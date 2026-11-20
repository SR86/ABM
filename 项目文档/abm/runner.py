"""Reproducible ABM experiment orchestration, stage checkpoints and metrics."""
import argparse
import hashlib
import json
import platform
import random
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .attacks import BadNetsAttack, BlendAttack
from .data import AttackTestDataset, TensorImageDataset, load_datasets
from .method import ABM, NBAB, compute_average_activation, evaluate_model, state_hash, train_student, train_teacher
from .models import build_model
from .monitoring import accuracy_probe

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
DEFAULTS = dict(
    dataset='cifar10', model_dataset=None, num_classes=10, image_size=32,
    data_root='data/cifar10', download=False, external_data=None,
    attack='badnets', target_label=0, poison_rate=.1,
    attack_trigger_size=2, attack_position=[27, 27], blend_alpha=.2, blend_pattern_mode='fixed',
    alpha=1000., delta=2.5, gamma=.8, activation_mode='label', control_rate=.1,
    control_trigger_size=2, isolation_rate=.01, nbab_epochs=10, nbab_lr=.01, nbab_l2_reduction='batch',
    teacher_epochs=50, teacher_lr=.1, teacher_step_size=2, teacher_lr_gamma=.1,
    student_epochs=20, student_lr=.01, momentum=.9, weight_decay=1e-4,
    batch_size=120, num_workers=0, seed=42, device='auto', cpu_threads=2,
    max_train_samples=None, max_test_samples=None, output_dir='results/cifar10_badnets', smoke_test=False,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, **payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def subset(dataset, count, seed):
    if count is None or count >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
    return Subset(dataset, indices)


def dataset_signature(dataset):
    digest = hashlib.sha256()
    for image, label in dataset:
        digest.update(image.contiguous().numpy().tobytes())
        digest.update(str(int(label)).encode() + b'\0')
    return digest.hexdigest()


def validate_config(c):
    if set(c) - set(DEFAULTS):
        raise ValueError(f'Unknown config keys: {sorted(set(c) - set(DEFAULTS))}')
    for key in ('nbab_epochs', 'teacher_epochs', 'student_epochs', 'batch_size', 'image_size', 'teacher_step_size',
                'cpu_threads', 'attack_trigger_size', 'control_trigger_size', 'num_classes'):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['num_classes'] < 2 or not isinstance(c['num_workers'], int) or c['num_workers'] < 0:
        raise ValueError('Need num_classes>=2 and num_workers>=0')
    if not isinstance(c['seed'], int) or not 0 <= c['seed'] < 2**32 - 100:
        raise ValueError('seed must be an integer in [0,2**32-100)')
    if not 0 <= c['target_label'] < c['num_classes']:
        raise ValueError('target_label outside class range')
    for key in ('control_rate', 'isolation_rate', 'teacher_lr_gamma'):
        if not 0 < c[key] <= 1:
            raise ValueError(f'{key} must be in (0,1]')
    if c['control_rate'] == 1:
        raise ValueError('control_rate must be <1 to preserve original classes')
    if not 0 <= c['poison_rate'] <= 1 or not 0 <= c['blend_alpha'] <= 1:
        raise ValueError('poison_rate/blend_alpha must be in [0,1]')
    for key in ('nbab_lr', 'teacher_lr', 'student_lr'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('alpha', 'delta', 'gamma', 'weight_decay', 'momentum'):
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if c['momentum'] >= 1:
        raise ValueError('momentum must be <1')
    for key in ('max_train_samples', 'max_test_samples'):
        if c[key] is not None and (not isinstance(c[key], int) or c[key] < 1):
            raise ValueError(f'{key} must be positive or null')
    if c['activation_mode'] not in ('label', 'class'):
        raise ValueError('activation_mode must be label or class')
    if c['nbab_l2_reduction'] not in ('batch', 'triggered'):
        raise ValueError('nbab_l2_reduction must be batch or triggered')
    if c['control_trigger_size'] > c['image_size']:
        raise ValueError('Control trigger is larger than the input')
    if c['dataset'] not in ('cifar10', 'gtsrb', 'imagenet_subset', 'external'):
        raise ValueError('Unknown dataset')
    if c['attack'] not in ('badnets', 'blend', 'external'):
        raise ValueError('Only badnets/blend implemented; use external for original attack exports')
    if (c['dataset'] == 'external') != (c['attack'] == 'external'):
        raise ValueError('dataset=external and attack=external must be paired')
    if (c['model_dataset'] or c['dataset']) not in ('cifar10', 'gtsrb', 'imagenet_subset', 'external'):
        raise ValueError('Unknown model_dataset')
    if c['dataset'] == 'cifar10' and (c['num_classes'] != 10 or c['image_size'] != 32):
        raise ValueError('Paper CIFAR10 profile requires 10 classes, 32x32 input')


def plot_results(metrics, path, label='Measured experiment'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle(label, fontsize=10)
    for ax, key, title in zip(axes, ('ca', 'asr'), ('Clean accuracy', 'Attack success rate')):
        bars = ax.bar(['Teacher', 'ABM'], [metrics['teacher'][key], metrics['abm'][key]], color=['#d16c61', '#399c9c'])
        ax.bar_label(bars, fmt='%.2f')
        ax.set(ylim=(0, 110), ylabel='Percent', title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(c, resume=False, prepare_only=False):
    validate_config(c)
    torch.set_num_threads(c['cpu_threads'])
    seed_all(c['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if c['device'] == 'auto' else torch.device(c['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA is unavailable in this PyTorch environment')
    if c['smoke_test']:
        g = torch.Generator().manual_seed(c['seed'])
        train = TensorImageDataset(torch.rand(40, 3, 32, 32, generator=g), torch.arange(40) % 10)
        test = TensorImageDataset(torch.rand(20, 3, 32, 32, generator=g), torch.arange(20) % 10)
        attack_test = None
    else:
        train, test, attack_test = load_datasets(c)
    if prepare_only:
        print(f'Data ready: train={len(train)}, clean_test={len(test)}', flush=True)
        return
    sizes = {'original_train': len(train), 'original_clean_test': len(test)}
    poison_indices = getattr(train, 'poison_indices', None)
    train = subset(train, c['max_train_samples'], c['seed'] + 10)
    test = subset(test, c['max_test_samples'], c['seed'] + 11)
    if attack_test is not None:
        attack_test = subset(attack_test, c['max_test_samples'], c['seed'] + 12)
        if isinstance(train, Subset) and poison_indices is not None:
            known = set(poison_indices)
            poison_indices = [i for i, old in enumerate(train.indices) if old in known]
    else:
        if c['attack'] == 'badnets':
            attack = BadNetsAttack(c['target_label'], c['poison_rate'], c['attack_trigger_size'], c['attack_position'], c['seed'])
        else:
            attack = BlendAttack(c['target_label'], c['poison_rate'], c['blend_alpha'], c['image_size'], c['seed'], c['blend_pattern_mode'])
        train, poison_indices = attack.poison_dataset(train)
        attack_test = AttackTestDataset(test, attack)
    output = Path(c['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        if not resume:
            raise ValueError(f'{output} already contains a run; use --resume or a new --output-dir')
        if json.loads(config_path.read_text(encoding='utf-8')) != c:
            raise ValueError('Resume configuration differs; use a new output directory')
    write_json(config_path, c)
    signature = {name: dataset_signature(data) for name, data in [('train', train), ('clean_test', test), ('attack_test', attack_test)]}
    signature['implementation'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted((ROOT / 'abm').glob('*.py'))}
    signature_path = output / 'data_signature.json'
    if resume and signature_path.exists():
        previous = json.loads(signature_path.read_text())
        if previous != signature:
            legacy = json.loads((ROOT / 'abm' / 'epochless_implementation.json').read_text())
            audited = json.loads((ROOT / 'abm' / 'pre_audit_implementation.json').read_text())
            if (previous.get('implementation') not in (legacy, audited) or
                    {k:v for k,v in previous.items() if k != 'implementation'} !=
                    {k:v for k,v in signature.items() if k != 'implementation'}):
                raise ValueError('Data or implementation changed; refusing to reuse checkpoints')
            write_json(output / 'epoch_checkpoint_migration.json',
                       dict(reason='Audited checkpoint-only upgrade; training equations unchanged', previous=previous))
    write_json(signature_path, signature)
    result = dict(status='running', validation_only=bool(c['smoke_test'] or c['max_train_samples'] or c['max_test_samples']),
                  synthetic_data=c['smoke_test'], config=c, started_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                                   device=str(device), cuda=torch.version.cuda),
                  data=dict(sizes, effective_train=len(train), effective_clean_test=len(test), asr_test=len(attack_test)))
    write_json(output / 'results.json', result)

    probe_train = subset(train, min(512, len(train)), c['seed'] + 80)
    probe_test = subset(test, min(1024, len(test)), c['seed'] + 81)
    last_heartbeat = [0.]

    def update_progress(stage, epoch, batch, total, force=False):
        if not force and time.monotonic() - last_heartbeat[0] < 30:
            return
        last_heartbeat[0] = time.monotonic()
        write_json(output / 'progress.json', dict(status='running', stage=stage, epoch=epoch,
                   batch=batch, batches=total, updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))

    def log(record):
        if record['stage'] in ('teacher', 'nbab'):
            active = teacher if record['stage'] == 'teacher' else verification
            train_ca = accuracy_probe(active, probe_train, device, c['batch_size'])
            test_ca = accuracy_probe(active, probe_test, device, c['batch_size'])
            diagnostic = dict(stage=record['stage'], epoch=record['epoch'], train_probe_accuracy=train_ca,
                              test_probe_accuracy=test_ca, gap_pp=train_ca-test_ca,
                              overfit_warning=train_ca-test_ca > 15, diagnostic_only=True,
                              updated_utc=datetime.now(timezone.utc).isoformat())
            with (output / 'generalization.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(diagnostic) + '\n')
        update_progress(record['stage'], record['epoch'], 0, 0, True)
        print(f"{record['stage']:8s} epoch {record['epoch']:3d} loss={record['loss']:.6f} lr={record['lr']:.3g}", flush=True)
        with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
    log.heartbeat = update_progress

    def loader(data, seed, shuffle=True):
        return DataLoader(data, batch_size=c['batch_size'], shuffle=shuffle, num_workers=c['num_workers'],
                          generator=torch.Generator().manual_seed(seed))

    architecture = c['model_dataset'] or c['dataset']
    try:
        nbab_path = output / 'nbab.pt'
        if resume and nbab_path.exists():
            ckpt = torch.load(nbab_path, weights_only=True, map_location='cpu')
            indices, candidates, nh = ckpt['isolated_indices'], ckpt['candidate_indices'], ckpt['history']
        else:
            seed_all(c['seed'] + 20)
            nbab = NBAB(c['num_classes'], c['control_trigger_size'], c['alpha'], c['control_rate'],
                        c['isolation_rate'], c['seed'], c['batch_size'], c['num_workers'], c['nbab_l2_reduction'])
            verification = build_model(architecture, c['num_classes'] + 1).to(device)
            nbab.create_isolated_dataset(train, verification, device, c['nbab_epochs'], c['nbab_lr'], c['momentum'], c['weight_decay'], log,
                                        checkpoint_path=output / 'nbab_epoch.pt')
            indices, candidates, nh = nbab.isolated_indices, nbab.candidate_indices, nbab.history
            save_checkpoint(nbab_path, model=verification.cpu().state_dict(), isolated_indices=indices,
                            candidate_indices=candidates, control_indices=nbab.control_indices, history=nh)
            del verification
        isolated = Subset(train, indices.tolist())
        purity = None if poison_indices is None else len(set(indices.tolist()) & set(poison_indices)) / len(indices)
        result['isolation'] = dict(candidates=len(candidates), selected=len(indices), rate=len(indices) / len(train),
                                   requested_rate=c['isolation_rate'], poison_purity_diagnostic_only=purity)
        write_json(output / 'isolation.json', dict(result['isolation'], indices=indices.tolist(), candidates=candidates.tolist()))
        print(f'NBAB: {len(candidates)} candidates; {len(indices)} selected', flush=True)
        teacher_path = output / 'teacher.pt'
        seed_all(c['seed'] + 30)
        teacher = build_model(architecture, c['num_classes']).to(device)
        if resume and teacher_path.exists():
            ckpt = torch.load(teacher_path, map_location='cpu', weights_only=True)
            teacher.load_state_dict(ckpt['model'])
            activation, th = ckpt['avg_activation'].to(device), ckpt['history']
        else:
            th = train_teacher(teacher, loader(train, c['seed'] + 30), device, c['teacher_epochs'], c['teacher_lr'],
                               c['momentum'], c['weight_decay'], c['teacher_step_size'], c['teacher_lr_gamma'], log,
                               checkpoint_path=output / 'teacher_epoch.pt')
            activation = compute_average_activation(teacher, loader(train, 0, False), device, c['activation_mode'])
            save_checkpoint(teacher_path, model=teacher.state_dict(), avg_activation=activation.cpu(), history=th)
        teacher.requires_grad_(False).eval()
        teacher_hash = state_hash(teacher)
        seed_all(c['seed'] + 40)
        student = build_model(architecture, c['num_classes']).to(device)
        student_path = output / 'student.pt'
        if resume and student_path.exists():
            ckpt = torch.load(student_path, map_location='cpu', weights_only=True)
            student.load_state_dict(ckpt['model'])
            sh = ckpt['history']
        else:
            sh = train_student(teacher, student, loader(isolated, c['seed'] + 40), device, c['student_epochs'],
                               c['student_lr'], c['momentum'], c['weight_decay'], log,
                               checkpoint_path=output / 'student_epoch.pt')
            save_checkpoint(student_path, model=student.state_dict(), history=sh)
        defended = ABM(teacher, student, activation, c['delta'], c['gamma']).to(device).eval()
        metrics = {name: {key: evaluate_model(model, data, device, c['batch_size'], c['num_workers'])
                          for key, data in [('ca', test), ('asr', attack_test)]}
                   for name, model in [('teacher', teacher), ('abm', defended)]}
        if state_hash(teacher) != teacher_hash:
            raise RuntimeError('Teacher changed during defense or evaluation')
        save_checkpoint(output / 'abm.pt', model=defended.state_dict())
        result.update(status='complete', metrics_percent=metrics, average_activation=activation.cpu().tolist(),
                      teacher_unchanged=True, teacher_state_sha256=teacher_hash, history=nh + th + sh)
        label = ('Synthetic smoke test - not paper metrics' if c['smoke_test'] else
                 'Subset validation - not paper metrics' if result['validation_only'] else 'Measured experiment')
        plot_results(metrics, output / 'metrics.png', label)
        write_json(output / 'results.json', result)
        print(json.dumps(metrics, indent=2), flush=True)
        print(f'Results: {output.resolve()}', flush=True)
        return result
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'results.json', result)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--smoke-test', action='store_true', help='Synthetic ResNet18 validation, not paper metrics')
    p.add_argument('--resume', action='store_true', help='Reuse complete stages; incomplete stages restart')
    p.add_argument('--prepare-data', action='store_true')
    p.add_argument('--download', action='store_true')
    keys = ('device', 'output_dir', 'max_train_samples', 'max_test_samples', 'nbab_epochs', 'teacher_epochs',
            'student_epochs', 'nbab_lr', 'teacher_lr', 'student_lr', 'seed')
    for key in keys:
        kind = float if key.endswith('_lr') else str if key in ('device', 'output_dir') else int
        p.add_argument('--' + key.replace('_', '-'), type=kind)
    args = p.parse_args(argv)
    c = DEFAULTS.copy()
    if args.config:
        c.update(json.loads(args.config.read_text(encoding='utf-8')))
    if args.smoke_test:
        c.update(smoke_test=True, dataset='cifar10', model_dataset=None, attack='badnets', num_classes=10, image_size=32,
                 output_dir='results/smoke', nbab_epochs=1, teacher_epochs=1, student_epochs=1, nbab_lr=.0001,
                 teacher_lr=.001, student_lr=.001, batch_size=8, isolation_rate=.2, control_rate=.2, poison_rate=.2,
                 device='cpu', max_train_samples=None, max_test_samples=None)
    for key in keys:
        if getattr(args, key) is not None:
            c[key] = getattr(args, key)
    if args.download:
        c['download'] = True
    for key in ('data_root', 'output_dir', 'external_data'):
        if c[key] is not None:
            c[key] = str((ROOT / c[key]).resolve())
    return run(c, args.resume, args.prepare_data)
