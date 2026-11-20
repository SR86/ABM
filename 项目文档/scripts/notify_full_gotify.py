"""One-shot Gotify notification after the existing full run terminates."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results' / 'full_run'


def notification(state):
    succeeded = state.get('status') == 'complete'
    lines = ['CIFAR-10 全量训练' + ('已完成。' if succeeded else '运行结束，存在失败，请检查日志。')]
    for attack, result in state.get('experiments', {}).items():
        lines.append(f"{attack}: {result.get('status', 'unknown')}")
        for model, values in (result.get('metrics_percent') or {}).items():
            lines.append(f"  {model}: CA={values['ca']:.2f}%, ASR={values['asr']:.2f}%")
    return {'title': 'ABM 全量训练完成' if succeeded else 'ABM 全量训练异常',
            'message': '\n'.join(lines), 'priority': 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Optional legacy JSON config')
    parser.add_argument('--url', default='http://127.0.0.1:8769')
    parser.add_argument('--check', action='store_true', help='Validate credentials and server without sending a message')
    parser.add_argument('--once', action='store_true', help='Check status once, sending only if finished')
    args = parser.parse_args()
    # Validate credentials before committing to monitoring; never print tokens.
    if args.config:
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    else:
        import keyring
        config = {'url': args.url, 'token': keyring.get_password('CodexQuotaNotifier.Gotify', 'app_token')}
    url = config['url'].rstrip('/')
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.netloc or parts.username or parts.password or parts.query:
        raise ValueError('Gotify url must be an HTTP(S) base URL without embedded credentials/query')
    token = config['token']
    if not isinstance(token, str) or not token.strip():
        raise ValueError('Gotify application token is empty')
    if args.check:
        request = urllib.request.Request(url + '/health')
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 200:
                raise RuntimeError('Gotify health check failed')
        print('Credential found in Windows Credential Manager; Gotify health check passed. No message sent.', flush=True)
        return
    print('Gotify completion watcher active; credentials loaded without writing token to disk.', flush=True)
    receipt = RUN / 'gotify_notification.json'
    while True:
        try:
            state = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            state = {}
        if state.get('status') in ('complete', 'failed'):
            run_id = state.get('updated_utc')
            if receipt.exists() and json.loads(receipt.read_text()).get('run_id') == run_id:
                return
            request = urllib.request.Request(url + '/message',
                data=json.dumps(notification(state), ensure_ascii=False).encode('utf-8'),
                headers={'X-Gotify-Key': token, 'Content-Type': 'application/json'}, method='POST')
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    payload = json.load(response)
                receipt.write_text(json.dumps({'run_id': run_id, 'message_id': payload.get('id'),
                    'sent_utc': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
                print('Gotify notification sent.', flush=True)
                return
            except Exception as exc:
                # Type only: do not leak server responses or credentials to logs.
                print('Gotify send failed: ' + type(exc).__name__, flush=True)
                if args.once:
                    raise SystemExit(1)
                time.sleep(60)
        elif args.once:
            print('Training has not finished; no notification sent.')
            return
        else:
            time.sleep(30)


if __name__ == '__main__':
    main()
