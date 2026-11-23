"""Gotify confirmation and terminal alerts for the running stability suite."""
import json
from pathlib import Path
import time
import urllib.request
import keyring
import argparse
from report_gotify import alive

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/nbab_stability'


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--description', default='2种攻击×3个训练种子×3档系数，每组20轮。')
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    receipt = RUN/'gotify_receipts.json'
    delivered = json.loads(receipt.read_text()) if receipt.exists() else {}
    while True:
        try:
            state = json.loads((RUN/'suite_status.json').read_text())
            status = state['status']
            if status == 'running' and not alive(int((RUN/'supervisor.pid').read_text().strip())):
                status = 'process_missing'
            key = 'started' if status == 'running' else status
            if key not in delivered:
                lines = [f'NBAB 实验：{status}；{args.description}',
                         f"当前任务：{state.get('current', state.get('current_seed', '?'))}",
                         '这是诊断实验，完成不代表论文复现成功。']
                for folder in sorted(RUN.iterdir()):
                    if not folder.is_dir():
                        continue
                    for result_path in sorted(folder.glob('*/result.json')):
                        result = json.loads(result_path.read_text())
                        h = result.get('identity_holdout', {})
                        purity, recall = h.get('candidate_poison_purity'), h.get('poison_recall')
                        metrics = '' if purity is None or recall is None else f'；纯度{purity:.2%}，召回{recall:.2%}'
                        lines.append(f"{folder.name}/{result_path.parent.name}: {result['status']}{metrics}")
                titles = {'running': 'NBAB 手机提醒已接通', 'complete': 'NBAB 实验已完成',
                          'failed': 'NBAB 实验存在失败', 'process_missing': 'NBAB 训练进程异常退出'}
                token = keyring.get_password('CodexQuotaNotifier.Gotify', 'app_token')
                if not token:
                    raise RuntimeError('Missing credential')
                payload = dict(title=titles.get(status, 'NBAB 训练状态'), message='\n'.join(lines), priority=5)
                request = urllib.request.Request('http://127.0.0.1:8769/message',
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                    headers={'X-Gotify-Key': token, 'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(request, timeout=20) as response:
                    message = json.load(response)
                delivered[key] = {'message_id': message.get('id'), 'sent_at': time.time()}
                temp = receipt.with_suffix('.tmp')
                temp.write_text(json.dumps(delivered, indent=2))
                temp.replace(receipt)
                print(f'Gotify sent: {key}; ID {message.get("id")}', flush=True)
            if status in ('complete', 'failed', 'process_missing'):
                return
        except Exception as exc:
            print('Notification retry pending: ' + type(exc).__name__, flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
