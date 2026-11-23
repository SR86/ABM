"""Queue author-recipe diagnostics after the live forward audit."""
from pathlib import Path
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from report_gotify import alive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/nbab_author_audit'
PRIOR = ROOT/'results/nbab_forward_audit'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'status':'running', 'current':'waiting_for_forward_audit', 'experiments':{}}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = OUT/'suite_status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(OUT/'suite_status.json')
    save()
    try:
        while True:
            previous = json.loads((PRIOR/'suite_status.json').read_text())
            pid = int((PRIOR/'supervisor.pid').read_text())
            if previous['status'] in ('complete','failed') and not alive(pid):
                break
            if previous['status'] == 'running' and not alive(pid):
                raise RuntimeError('Prior audit exited without terminal status; inspect before launching')
            time.sleep(30)
        for mode in ('overlap_allowed', 'oracle_disjoint'):
            state['current'] = mode
            save()
            cmd = [sys.executable, str(ROOT/'scripts/nbab_author_recipe.py'), '--output', str(OUT/mode)]
            if mode == 'oracle_disjoint':
                cmd.append('--oracle-disjoint-control')
            with (OUT/f'{mode}.log').open('a') as log:
                run = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result = json.loads((OUT/mode/'status.json').read_text()) if (OUT/mode/'status.json').exists() else {}
            state['experiments'][mode] = {'returncode':run.returncode,'status':result.get('status','failed')}
            save()
        state['status'] = 'complete' if all(x['returncode']==0 and x['status']=='complete' for x in state['experiments'].values()) else 'failed'
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()

if __name__ == '__main__':
    main()
