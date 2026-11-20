"""Hourly local training reports plus prompt terminal/process-failure alerts."""
import ctypes
import argparse
import json
from pathlib import Path
import time
import urllib.request
import keyring

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/full_run'


def alive(pid):
    dll = ctypes.WinDLL('kernel32', use_last_error=True)
    dll.OpenProcess.restype = ctypes.c_void_p
    dll.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    dll.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = dll.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(dll.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        dll.CloseHandle(handle)


def report(state):
    attack = state.get('current_attack', 'unknown')
    status = state.get('status', 'unknown')
    diagnostic = state.get('kind') == 'nbab_diagnostics'
    folder = RUN / state['current_experiment'] if diagnostic else ROOT / 'results' / f'cifar10_{attack}'
    lines = [f'ABM {"NBAB短实验" if diagnostic else "全量实验"}：{status}；当前攻击：{attack}']
    if diagnostic:
        lines.append('当前配置：' + state['current_experiment'])
    progress = folder / 'progress.json'
    if progress.exists():
        p = json.loads(progress.read_text())
        limits = {'nbab':10, 'teacher':50, 'student':20}
        lines.append(f"阶段 {p['stage']}，轮次 {p['epoch']}/{p.get('epochs',limits.get(p['stage'],'?'))}，批次 {p['batch']}/{p['batches']}")
        lines.append('最近心跳 UTC：' + p['updated_utc'])
    diagnostic = folder / 'generalization.jsonl'
    if diagnostic.exists():
        d = json.loads(diagnostic.read_text().splitlines()[-1])
        lines.append(f"诊断：训练/测试探针准确率 {d['train_probe_accuracy']:.2f}%/{d['test_probe_accuracy']:.2f}%，差距 {d['gap_pp']:.2f}pp；过拟合告警 {d['overfit_warning']}")
    for name, item in state.get('experiments', {}).items():
        lines.append(f"{name}: {item['status']}")
        for model, values in (item.get('metrics_percent') or {}).items():
            lines.append(f"{model}: CA {values['ca']:.2f}%, ASR {values['asr']:.2f}%")
        if item.get('error'):
            lines.append(str(item['error'])[:300])
        if item.get('identity_holdout'):
            heldout = item['identity_holdout']
            lines.append(f"独立身份验证：干净误报率 {heldout['clean_false_positive_rate']:.2%}；候选纯度 {heldout['candidate_poison_purity']}")
    if status == 'running':
        pidfile = RUN / 'supervisor.pid'
        if not pidfile.exists() or not alive(int(pidfile.read_text().strip())):
            status = 'process_missing'
            lines.append('异常：训练监督进程已退出，状态文件可能过期，请检查并恢复。')
        elif any(item.get('status') == 'failed' for item in state.get('experiments', {}).values()):
            status = 'experiment_failed'
    title = {'complete':'ABM 全量训练成功', 'failed':'ABM 全量训练失败',
             'experiment_failed':'ABM 单项训练失败，其他实验继续',
             'process_missing':'ABM 训练进程异常退出', 'paused':'ABM 训练已暂停'}.get(status, 'ABM 每小时训练进度')
    if diagnostic and status == 'complete':
        title = 'NBAB 短实验完成（非论文复现验收）'
    elif diagnostic and status == 'failed':
        title = 'NBAB 短实验结束，部分配置失败'
    return status, dict(title=title, message='\n'.join(lines), priority=5)


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    receipt = RUN / 'gotify_progress_receipt.json'
    delivered = json.loads(receipt.read_text()) if receipt.exists() else {}
    print('Hourly Gotify reporter active (3600 seconds); terminal/process alerts enabled.', flush=True)
    while True:
        try:
            state = json.loads((RUN / 'status.json').read_text())
            status, payload = report(state)
            terminal = status in ('complete', 'failed', 'process_missing', 'experiment_failed')
            key = (f"{status}:{state.get('updated_utc')}" if status in ('complete', 'failed') else
                   f"{status}:{int(time.time() // 3600)}")
            if delivered.get('key') != key and (terminal or time.time()-delivered.get('sent_at', 0) >= 3600):
                token = keyring.get_password('CodexQuotaNotifier.Gotify', 'app_token')
                if not token:
                    raise RuntimeError('Gotify credential missing')
                request = urllib.request.Request('http://127.0.0.1:8769/message',
                    data=json.dumps(payload, ensure_ascii=False).encode(),
                    headers={'X-Gotify-Key':token, 'Content-Type':'application/json'}, method='POST')
                with urllib.request.urlopen(request, timeout=20) as response:
                    message = json.load(response)
                delivered = dict(key=key, sent_at=time.time(), message_id=message.get('id'))
                temp = receipt.with_suffix('.tmp')
                temp.write_text(json.dumps(delivered))
                temp.replace(receipt)
                print(f'Gotify sent: {status}; message ID {message.get("id")}', flush=True)
            if status in ('complete', 'failed'):
                return
        except Exception as exc:
            print('Report retry pending: ' + type(exc).__name__, flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
