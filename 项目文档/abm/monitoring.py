"""Diagnostic evaluation with no model selection and no training RNG changes."""
import time
from datetime import datetime, timezone
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def accuracy_probe(model, dataset, device, batch_size=120):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        correct = count = 0
        devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            loader = DataLoader(dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(0))
            for images, labels in loader:
                prediction = model(images.to(device)).argmax(1)
                correct += prediction.eq(labels.to(device)).sum().item()
                count += len(labels)
        return 100. * correct / count
    finally:
        for module, mode in modes:
            module.training = mode


def heartbeat(log, stage, epoch, batch, total):
    if log is not None and hasattr(log, 'heartbeat'):
        log.heartbeat(stage, epoch, batch, total)
