"""Summarize existing evidence without selecting epochs or changing training."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def main():
    output = {'updated_utc':datetime.now(timezone.utc).isoformat(),
              'author_teacher_schedule':{'source':'tmp/author_abm/Util.py:458',
                  'zero_based_epochs_0_1':0.1, 'epochs_2_19':0.01, 'epochs_20_onward':0.0001,
                  'script_total_epochs':30, 'interpretation':'Author schedule differs from paper literal StepLR; causal comparison pending.'},
              'teacher_schedules':{},
              'epoch_evaluation_trajectory_checks':[], 'paired_forward_comparisons':[]}
    for attack in ('badnets','blend'):
        result = read(ROOT/f'results/cifar10_{attack}/results.json')
        output['teacher_schedules'][attack] = {
            'final_metrics':result['metrics_percent']['teacher'],
            'selected_history':[r for r in result['history'] if r['stage']=='teacher' and r['epoch'] in (1,2,4,6,10,20,50)],
            'interpretation':'Paper literally specifies 10x decay every 2 epochs. Very small later LR is evidence of a limitation, not proof of a typo or permission to relabel a changed schedule as exact reproduction.'}
        for alpha in ('0','0.1','1000'):
            name = f'{attack}_logits_alpha_{alpha}_sgd'
            joint = ROOT/'results/nbab_forward_audit/joint'/name/'result.json'
            prior = ROOT/'results/nbab_strength_scan'/name/'result.json'
            separate = ROOT/'results/nbab_forward_audit/separate'/name/'result.json'
            if joint.exists() and prior.exists():
                a,b = read(joint),read(prior)
                n=min(len(a['history']),len(b['history']))
                differences = [abs(a['history'][i][key]-b['history'][i][key]) for i in range(n) for key in ('ce','regularizer')]
                output['epoch_evaluation_trajectory_checks'].append({'job':name,'compared_epochs':n,'max_loss_difference':max(differences),
                    'meaning':'Same first epochs with added evaluation; numerical loss check, not proof of all model state equality.'})
            if joint.exists() and separate.exists():
                a,b = read(joint),read(separate)
                output['paired_forward_comparisons'].append({'job':name,
                    'joint_final':a.get('identity_holdout'), 'separate_final':b.get('identity_holdout'),
                    'joint_history':a.get('history'), 'separate_history':b.get('history')})
    destination=ROOT/'results/nbab_forward_audit/evidence_summary.json'
    destination.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k != 'paired_forward_comparisons'},indent=2))

if __name__ == '__main__':
    main()
