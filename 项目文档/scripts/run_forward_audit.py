"""Predeclared paired-forward comparison; sequential CPU execution."""
from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/nbab_forward_audit'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'experiments': {}}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = OUT/'suite_status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(OUT/'suite_status.json')
    save()
    try:
        for mode in ('joint', 'separate'):
            state['current'] = mode
            save()
            cmd = [sys.executable, str(ROOT/'scripts/nbab_diagnostics.py'), '--strength-scan',
                   '--coefficients', '0', '0.1', '1000', '--epochs', '10',
                   '--evaluate-each-epoch', '--output', str(OUT/mode)]
            if mode == 'separate':
                cmd.append('--separate-forward')
            with (OUT/f'{mode}.log').open('a') as log:
                run = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result = json.loads((OUT/mode/'status.json').read_text()) if (OUT/mode/'status.json').exists() else {}
            state['experiments'][mode] = {'returncode':run.returncode, 'status':result.get('status','failed')}
            save()
        state['status'] = 'complete' if all(x['returncode']==0 and x['status']=='complete' for x in state['experiments'].values()) else 'failed'
    except Exception as exc:
        state.update(status='failed', error=type(exc).__name__)
        raise
    finally:
        save()

if __name__ == '__main__':
    main()
