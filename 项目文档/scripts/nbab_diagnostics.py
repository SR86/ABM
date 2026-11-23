"""Predeclared NBAB ablations on CIFAR training identities, never test-set tuning."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from author_compat.resnet import resnet
from abm.attacks import BadNetsAttack, BlendAttack
from abm.method import _ControlledDataset
from abm.checkpoints import restore_epoch, save_epoch

VARIANTS = [
    ('no_regularizer_sgd', 'logits', 'none', 'sgd'),
    ('logits_mse_sgd', 'logits', 'mse', 'sgd'),
    ('pooled_mse_sgd', 'pooled', 'mse', 'sgd'),
    ('stage3_mse_sgd', 'stage3', 'mse', 'sgd'),
    ('stage3_sum2000_sgd', 'stage3', 'sum2000', 'sgd'),
    ('stage3_sum2000_adam', 'stage3', 'sum2000', 'adam'),
    ('logits_mse_adam', 'logits', 'mse', 'adam'),
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def regularizer(left, right, mask, kind, strength=1000.):
    if kind == 'none' or not mask.any():
        return (left.sum() + right.sum()) * 0
    diff = left[mask] - right[mask]
    if kind == 'mse':
        return strength * mask.float().mean() * diff.square().mean()
    if kind == 'sum2000':
        return diff.square().sum() / 2000
    raise ValueError(kind)


@torch.no_grad()
def evaluate(model, data, poison_indices, control_indices, batch_size):
    model.eval()
    predictions = []
    clean_correct = count = 0
    with torch.random.fork_rng():
        for images, labels in DataLoader(data, batch_size=batch_size, generator=torch.Generator().manual_seed(99)):
            logits = model(images)[-1]
            clean_correct += logits.argmax(1).eq(labels).sum().item()
            count += len(labels)
            triggered = images.clone()
            triggered[:, :, :2, :2] = 1
            predictions.extend(model(triggered)[-1].argmax(1).tolist())
    poison = set(map(int, poison_indices))
    control = set(map(int, control_indices))
    candidate = {i for i, label in enumerate(predictions) if label != 10}
    clean = set(range(len(data))) - poison
    def group(indices):
        return dict(count=len(indices), newclass_rate=None if not indices else len(indices-candidate)/len(indices))
    return dict(original_label_accuracy=100*clean_correct/count,
        candidate_count=len(candidate), candidate_poison_purity=None if not candidate else len(candidate & poison)/len(candidate),
        poison_recall=len(candidate & poison)/len(poison) if poison else None,
        clean_false_positive_rate=len(candidate & clean)/len(clean),
        clean_control=group(clean & control), clean_uncontrolled=group(clean-control),
        poison_control=group(poison & control), poison_uncontrolled=group(poison-control))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-size', type=int, default=8000)
    parser.add_argument('--holdout-size', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--separate-forward', action='store_true')
    parser.add_argument('--evaluate-each-epoch', action='store_true')
    parser.add_argument('--coefficients', type=float, nargs='+', help='Predeclared alpha grid; requires --strength-scan')
    parser.add_argument('--strength-scan', action='store_true', help='Fixed logits/SGD; vary only alpha over a predefined grid')
    parser.add_argument('--output', type=Path, default=ROOT/'results/nbab_diagnostics')
    args = parser.parse_args()
    if args.train_size < 20 or args.holdout_size < 20 or args.train_size+args.holdout_size > 50000 or args.epochs < 1:
        raise ValueError('Invalid diagnostic sizes/epochs')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.coefficients and not args.strength_scan:
        parser.error('--coefficients requires --strength-scan')
    coefficients = tuple(args.coefficients) if args.coefficients else (0., .01, .1, 1., 10., 100., 1000.)
    if any(not math.isfinite(v) or v < 0 for v in coefficients) or len(set(coefficients)) != len(coefficients):
        parser.error('Coefficients must be unique, finite and nonnegative')
    variants = [(f'logits_alpha_{value:g}_sgd', 'logits', 'mse', 'sgd') for value in coefficients] if args.strength_scan else VARIANTS
    strengths = {item[0]:value for item,value in zip(variants, coefficients)} if args.strength_scan else {item[0]:1000. for item in variants}
    config = dict(train_size=args.train_size, holdout_size=args.holdout_size, epochs=args.epochs,
                  batch_size=120, seed=args.seed, lr=.01, control_rate=.1, poison_rate=.1,
                  architecture='NAD-compatible CIFAR ResNet20 candidate, not confirmed ABM original',
                  variants=variants, regularization_strengths=strengths, test_set_used=False, screening_only=True,
                  python=sys.version, torch=str(torch.__version__), numpy=np.__version__)
    config_path = output/'config.json'
    config.update(separate_forward=args.separate_forward, evaluate_each_epoch=args.evaluate_each_epoch)
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Existing diagnostic configuration differs')
    write(config_path, config)
    fingerprint = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__), ROOT/'author_compat/resnet.py', ROOT/'abm/checkpoints.py',
                             ROOT/'abm/attacks.py', ROOT/'abm/data.py', ROOT/'abm/method.py']}
    signature_path = output/'source_signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != fingerprint:
        raise ValueError('Diagnostic source changed; choose another output directory')
    write(signature_path, fingerprint)
    raw = CIFAR10(str(ROOT/'data/cifar10'), train=True, download=False, transform=ToTensor())
    order = torch.randperm(len(raw), generator=torch.Generator().manual_seed(73)).tolist()
    training = Subset(raw, order[:args.train_size])
    heldout = Subset(raw, order[args.train_size:args.train_size+args.holdout_size])
    write(output/'identity_split.json', dict(train=training.indices, holdout=heldout.indices))
    state = dict(status='running', kind='nbab_diagnostics', experiments={},
                 updated_utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid())
    (output/'supervisor.pid').write_text(str(os.getpid()))
    write(output/'status.json', state)
    for attack_name in ('badnets', 'blend'):
        attack_cls = BadNetsAttack if attack_name == 'badnets' else BlendAttack
        attack = attack_cls(seed=42)
        train, poison = attack.poison_dataset(training)
        # Holdout attack realization uses a distinct seed and identities.
        holdout_attack = attack_cls(seed=43)
        if attack_name == 'blend':
            holdout_attack.trigger = attack.trigger.clone()  # fixed attack pattern is common to both splits
        validation, holdout_poison = holdout_attack.poison_dataset(heldout)
        for name, layer, reduction, optimizer_name in variants:
            job = f'{attack_name}_{name}'
            folder = output/job
            folder.mkdir(exist_ok=True)
            state.update(current_attack=attack_name, current_experiment=job, updated_utc=datetime.now(timezone.utc).isoformat())
            completed = folder/'result.json'
            if completed.exists():
                state['experiments'][job] = json.loads(completed.read_text())
                continue
            state['experiments'][job] = dict(status='running')
            write(output/'status.json', state)
            try:
                seed(args.seed)
                model = resnet(depth=20, num_classes=11, near_mode=layer)
                control = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[:int(.1*len(train))]
                controlled = _ControlledDataset(train, control, 2, 10)
                loader = DataLoader(controlled, batch_size=120, shuffle=True, generator=torch.Generator().manual_seed(args.seed+2))
                optimizer = (torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
                             if optimizer_name == 'sgd' else torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-4))
                history = restore_epoch(folder/'epoch.pt', model, optimizer, loader)
                last_ping = 0
                for epoch in range(len(history), args.epochs):
                    model.train()
                    ce_sum = reg_sum = count = 0
                    for batch, (images, labels, original, mask) in enumerate(loader, 1):
                        if time.monotonic()-last_ping > 30:
                            write(folder/'progress.json', dict(status='running', stage='nbab', epoch=epoch+1, epochs=args.epochs,
                                batch=batch, batches=len(loader), pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat()))
                            last_ping = time.monotonic()
                        optimizer.zero_grad(set_to_none=True)
                        if args.separate_forward:
                            result = model(images)
                            paired = model(original)
                            logits, original_logits = result[-1], paired[-1]
                            near, reference = result[2], paired[2]
                        else:
                            result = model(torch.cat((images, original)))
                            logits, original_logits = result[-1].chunk(2)
                            near, reference = result[2].chunk(2)
                        ce = F.cross_entropy(logits, labels)
                        penalty = regularizer(near, reference, mask, reduction, strengths[name])
                        loss = ce + penalty
                        if not torch.isfinite(loss):
                            raise RuntimeError('Non-finite diagnostic loss; configuration recorded as failed')
                        if epoch == 0 and batch == 1:
                            def grad_norm(objective):
                                values = torch.autograd.grad(objective, model.parameters(), retain_graph=True, allow_unused=True)
                                return math.sqrt(sum(float(v.square().sum()) for v in values if v is not None))
                            write(folder/'initial_loss_scale.json', dict(ce=ce.item(), regularizer=penalty.item(),
                                ce_gradient_norm=grad_norm(ce), regularizer_gradient_norm=grad_norm(penalty), near_shape=list(near.shape)))
                        loss.backward()
                        optimizer.step()
                        ce_sum += ce.item()*len(images)
                        reg_sum += penalty.item()*len(images)
                        count += len(images)
                    row = dict(epoch=epoch+1, ce=ce_sum/count, regularizer=reg_sum/count)
                    if args.evaluate_each_epoch:
                        row['identity_holdout'] = evaluate(model, validation, holdout_poison, [], 120)
                    history.append(row)
                    save_epoch(folder/'epoch.pt', model, optimizer, history, loader)
                    with (folder/'history.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    print(job, row, flush=True)
                result = dict(status='complete', layer=layer, reduction=reduction, optimizer=optimizer_name, alpha=strengths[name],
                    train_diagnostic=evaluate(model, train, poison, control.tolist(), 120),
                    identity_holdout=evaluate(model, validation, holdout_poison, [], 120), history=history,
                    note='Small identity-holdout screening; not full paper reproduction; poison truth diagnostic only')
            except Exception as exc:
                result = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
                print(job, result, flush=True)
            write(completed, result)
            state['experiments'][job] = result
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            write(output/'status.json', state)
    state['status'] = 'complete' if all(v['status']=='complete' for v in state['experiments'].values()) else 'failed'
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    write(output/'status.json', state)
    lines = ['# NBAB 单变量短实验结果', '',
        '固定 NAD 兼容候选 ResNet20；仅使用 CIFAR 训练身份分割。不是原全量 ResNet18 实验，也不是论文数值复现。', '',
        '|配置|状态|独立干净身份触发成功率|独立候选纯度|独立投毒召回率|',
        '|---|---|---:|---:|---:|']
    def percent(value):
        return '无候选/不适用' if value is None else f'{100*value:.2f}%'
    for job, result in state['experiments'].items():
        h = result.get('identity_holdout', {})
        success = h.get('clean_uncontrolled', {}).get('newclass_rate')
        lines.append(f"|{job}|{result['status']}|{percent(success)}|{percent(h.get('candidate_poison_purity'))}|{percent(h.get('poison_recall'))}|")
    lines += ['', '独立干净身份触发成功率、候选纯度与投毒召回率需联合判断；全部预测新增类可能导致零候选，不能算防御成功。',
              'poison truth 仅事后评估；未用于训练损失或候选决策。短实验只用于定位问题，不能据此声称论文指标达成。']
    (output/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('NBAB diagnostic suite finished:',state['status'],flush=True)


if __name__ == '__main__':
    main()
