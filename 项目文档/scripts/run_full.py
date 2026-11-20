"""Run both full CIFAR10 experiments sequentially and retain durable logs."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'full_run'


def save(state):
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temp = OUT / 'status.tmp'
    temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temp.replace(OUT / 'status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-incomplete', action='store_true', help='Keep completed experiment results and resume only unfinished attacks')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'experiments': {}}
    save(state)
    for attack in ('badnets', 'blend'):
        completed_path = ROOT / 'results' / f'cifar10_{attack}' / 'results.json'
        if args.continue_incomplete and completed_path.exists():
            completed = json.loads(completed_path.read_text(encoding='utf-8'))
            if completed.get('status') == 'complete' and completed.get('metrics_percent'):
                state['experiments'][attack] = dict(status='complete', returncode=0,
                    metrics_percent=completed['metrics_percent'], error=None)
                save(state)
                print(f'Keeping completed {attack} results; no retraining or reevaluation.', flush=True)
                continue
        state['current_attack'] = attack
        state['experiments'][attack] = {'status': 'running'}
        save(state)
        command = [sys.executable, '-u', '-X', 'utf8', str(ROOT / 'abm_experiment.py'),
                   '--config', str(ROOT / 'configs' / f'cifar10_{attack}.json')]
        if (ROOT / 'results' / f'cifar10_{attack}' / 'config.json').exists():
            command.append('--resume')
        with (OUT / f'{attack}.log').open('a', encoding='utf-8') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            except Exception as exc:
                state['experiments'][attack] = {'status':'failed', 'error':type(exc).__name__}
                state['status'] = 'failed'
                save(state)
                raise
        path = ROOT / 'results' / f'cifar10_{attack}' / 'results.json'
        payload = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        state['experiments'][attack] = {'status': payload.get('status', 'failed') if result.returncode == 0 else 'failed',
                                        'returncode': result.returncode,
                                        'metrics_percent': payload.get('metrics_percent'),
                                        'error': payload.get('error')}
        save(state)
    state['status'] = 'complete' if all(v['status'] == 'complete' for v in state['experiments'].values()) else 'failed'
    save(state)


if __name__ == '__main__':
    main()
