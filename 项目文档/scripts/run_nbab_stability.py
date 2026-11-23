"""Run the predeclared stability suite sequentially without CPU oversubscription."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/nbab_stability'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'seeds': {}}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = OUTPUT / 'suite_status.tmp'
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(OUTPUT / 'suite_status.json')
    save()
    try:
        for seed in (42, 123, 2026):
            state['current_seed'] = seed
            save()
            folder = OUTPUT / f'seed_{seed}'
            with (OUTPUT / f'seed_{seed}.log').open('a', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, str(ROOT/'scripts/nbab_diagnostics.py'),
                    '--strength-scan', '--coefficients', '0', '0.1', '1', '--epochs', '20',
                    '--seed', str(seed), '--output', str(folder)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status_file = folder/'status.json'
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            state['seeds'][str(seed)] = {'returncode': result.returncode, 'status': status.get('status', 'failed')}
            save()
        state['status'] = 'complete' if all(v['returncode'] == 0 and v['status'] == 'complete' for v in state['seeds'].values()) else 'failed'
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
