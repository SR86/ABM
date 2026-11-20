"""Equations (2)-(4), with all conventions explicit in configuration."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from .checkpoints import restore_epoch, save_epoch
from .monitoring import heartbeat


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def nbab_loss(logits, labels, original_logits, triggered_mask, alpha, l2_reduction='batch'):
    """Mean over Eq. (2)'s per-sample branches; L2 is mean over classes.

    Default weights the control subset by its fraction of the whole batch.
    'triggered' instead averages the regularizer only over control samples.
    Both paired branches receive gradients. No detaching or oracle labels.
    """
    if l2_reduction not in ('batch', 'triggered'):
        raise ValueError('l2_reduction must be batch or triggered')
    loss = F.cross_entropy(logits, labels)
    if triggered_mask.any():
        weight = triggered_mask.float().mean() if l2_reduction == 'batch' else 1.
        loss = loss + alpha * weight * F.mse_loss(logits[triggered_mask], original_logits[triggered_mask])
    return loss


def abm_logits(teacher_logits, student_logits, avg_activation, delta=2.5, gamma=.8):
    if teacher_logits.shape != student_logits.shape:
        raise ValueError('Teacher and student logits must have identical [batch, classes] shapes.')
    activation = torch.as_tensor(avg_activation, device=student_logits.device, dtype=student_logits.dtype)
    if activation.ndim > 1 or (activation.ndim == 1 and activation.numel() not in (1, student_logits.shape[-1])):
        raise ValueError('A must be a scalar or a vector with one entry per class.')
    return teacher_logits - delta * F.relu(student_logits - gamma * activation)


class ABM(nn.Module):
    def __init__(self, teacher, student, avg_activation, delta=2.5, gamma=.8):
        super().__init__()
        self.teacher = teacher.requires_grad_(False).eval()
        self.student = student
        self.register_buffer('avg_activation', torch.as_tensor(avg_activation).detach().clone())
        self.delta, self.gamma = delta, gamma

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        return abm_logits(teacher_logits, self.student(x), self.avg_activation, self.delta, self.gamma)


@torch.no_grad()
def compute_average_activation(model, loader, device, mode='label'):
    """Final-model eval statistic on ALL untrusted training data.

    label: scalar mean f(x)[y] as in author Util.py; class: per-output mean.
    No poison annotations or clean holdout are used.
    """
    if mode not in ('label', 'class'):
        raise ValueError('activation_mode must be label or class')
    model.eval()
    total, count = None, 0
    for images, labels in loader:
        logits = model(images.to(device))
        values = logits.gather(1, labels.to(device).view(-1, 1)).sum() if mode == 'label' else logits.sum(0)
        total = values if total is None else total + values
        count += len(images)
    if count == 0:
        raise ValueError('Cannot calculate A from an empty training dataset.')
    return (total / count).detach()


def _emit(log, record):
    if log is not None:
        log(record)


def train_teacher(model, loader, device, epochs=50, lr=.1, momentum=.9,
                  weight_decay=1e-4, step_size=2, lr_gamma=.1, log=None, checkpoint_path=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_gamma)
    history = restore_epoch(checkpoint_path, model, optimizer, loader, scheduler)
    for epoch in range(len(history), epochs):
        model.train()
        loss_sum, count = 0., 0
        current_lr = optimizer.param_groups[0]['lr']
        for batch, (images, labels) in enumerate(loader, 1):
            heartbeat(log, 'teacher', epoch + 1, batch, len(loader))
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite teacher loss; inspect the configuration.')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
            count += len(images)
        if not count:
            raise ValueError('Empty teacher dataset.')
        record = dict(stage='teacher', epoch=epoch + 1, loss=loss_sum / count, lr=current_lr)
        history.append(record)
        _emit(log, record)
        scheduler.step()
        save_epoch(checkpoint_path, model, optimizer, history, loader, scheduler)
    return history


def train_student(teacher, student, loader, device, epochs=20, lr=.01,
                  momentum=.9, weight_decay=1e-4, log=None, checkpoint_path=None):
    if len(loader.dataset) == 0:
        raise ValueError('NBAB isolated no samples. Student training cannot proceed.')
    teacher.requires_grad_(False).eval()
    teacher.zero_grad(set_to_none=True)
    before = state_hash(teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    history = restore_epoch(checkpoint_path, student, optimizer, loader)
    for epoch in range(len(history), epochs):
        student.train()
        loss_sum, count = 0., 0
        for batch, (images, _) in enumerate(loader, 1):
            heartbeat(log, 'student', epoch + 1, batch, len(loader))
            images = images.to(device)
            with torch.no_grad():
                target_logits = teacher(images)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(student(images), target_logits)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite student loss; inspect the configuration.')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
            count += len(images)
        record = dict(stage='student', epoch=epoch + 1, loss=loss_sum / count, lr=lr)
        history.append(record)
        _emit(log, record)
        save_epoch(checkpoint_path, student, optimizer, history, loader)
    if state_hash(teacher) != before:
        raise RuntimeError('Non-invasive invariant violated: teacher state changed.')
    return history


class _ControlledDataset(Dataset):
    def __init__(self, dataset, indices, trigger_size, new_label):
        self.dataset, self.indices = dataset, set(map(int, indices))
        self.trigger_size, self.new_label = trigger_size, new_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original, label = self.dataset[index]
        image = original.clone()
        triggered = index in self.indices
        if triggered:
            image[:, :self.trigger_size, :self.trigger_size] = 1.
            label = self.new_label
        return image, label, original, triggered


class NBAB:
    def __init__(self, num_classes=10, trigger_size=2, alpha=1000., control_rate=.1,
                  isolation_rate=.01, seed=42, batch_size=120, num_workers=0, l2_reduction='batch'):
        self.num_classes, self.trigger_size, self.alpha = num_classes, trigger_size, alpha
        self.control_rate, self.isolation_rate, self.seed = control_rate, isolation_rate, seed
        self.batch_size, self.num_workers = batch_size, num_workers
        self.l2_reduction = l2_reduction
        self.candidate_indices = torch.empty(0, dtype=torch.long)
        self.isolated_indices = torch.empty(0, dtype=torch.long)
        self.history = []

    def create_isolated_dataset(self, dataset, verification_model, device='cpu', epochs=10,
                                lr=.01, momentum=.9, weight_decay=1e-4, log=None, checkpoint_path=None):
        """Train C+1 verification model on D2; return original D1 subset.

        Sample identity is the only information available to this filter;
        attack poison_indices are deliberately never inspected here.
        """
        n = len(dataset)
        if n == 0 or not (0 < self.control_rate < 1) or not (0 < self.isolation_rate <= 1):
            raise ValueError('NBAB needs nonempty data, control_rate in (0,1), isolation_rate in (0,1].')
        generator = torch.Generator().manual_seed(self.seed)
        self.control_indices = torch.randperm(n, generator=generator)[:max(1, int(n * self.control_rate))]
        controlled = _ControlledDataset(dataset, self.control_indices, self.trigger_size, self.num_classes)
        loader = DataLoader(controlled, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, generator=generator)
        model = verification_model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.history = restore_epoch(checkpoint_path, model, optimizer, loader)
        for epoch in range(len(self.history), epochs):
            model.train()
            loss_sum, count = 0., 0
            for batch, (images, labels, originals, mask) in enumerate(loader, 1):
                heartbeat(log, 'nbab', epoch + 1, batch, len(loader))
                images, labels, originals, mask = [x.to(device) for x in (images, labels, originals, mask)]
                optimizer.zero_grad(set_to_none=True)
                # Joint forward shares a consistent BN context for paired samples.
                combined = model(torch.cat((images, originals), dim=0))
                logits, original_logits = combined.chunk(2)
                loss = nbab_loss(logits, labels, original_logits, mask, self.alpha, self.l2_reduction)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite NBAB loss; lower nbab_lr or review L2 scaling explicitly.')
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(images)
                count += len(images)
            record = dict(stage='nbab', epoch=epoch + 1, loss=loss_sum / count, lr=lr)
            self.history.append(record)
            _emit(log, record)
            save_epoch(checkpoint_path, model, optimizer, self.history, loader)
        model.eval()
        candidates, offset = [], 0
        with torch.no_grad():
            for images, _ in DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers):
                triggered = images.to(device).clone()
                triggered[:, :, :self.trigger_size, :self.trigger_size] = 1.
                keep = model(triggered).argmax(1).ne(self.num_classes).nonzero().flatten().cpu()
                candidates.extend((keep + offset).tolist())
                offset += len(images)
        self.candidate_indices = torch.tensor(candidates, dtype=torch.long)
        requested = max(1, math.floor(n * self.isolation_rate))
        if not candidates:
            raise ValueError('NBAB isolated zero candidates; no oracle fallback is permitted. Inspect verification training.')
        order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(self.seed + 1))
        self.isolated_indices = self.candidate_indices[order[:requested]]
        return Subset(dataset, self.isolated_indices.tolist())


@torch.no_grad()
def evaluate_model(model, dataset, device='cpu', batch_size=120, num_workers=0):
    model.eval()
    correct, total = 0, 0
    for images, labels in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        prediction = model(images.to(device)).argmax(1)
        correct += prediction.eq(labels.to(device)).sum().item()
        total += len(images)
    if total == 0:
        raise ValueError('Evaluation dataset is empty (ASR requires non-target examples).')
    return 100. * correct / total
