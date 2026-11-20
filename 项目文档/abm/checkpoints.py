"""Atomic epoch snapshots; serialization never consumes training randomness."""
from pathlib import Path
import os
import random
import shutil
import numpy as np
import torch


def save_epoch(path, model, optimizer, history, loader, scheduler=None):
    if path is None:
        return
    path = Path(path)
    state = np.random.get_state()
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                   scheduler=None if scheduler is None else scheduler.state_dict(),
                   python_rng=random.getstate(), numpy_rng=(state[0], state[1].tolist(), *state[2:]),
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                   loader_rng=loader.generator.get_state() if loader.generator is not None else None)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        previous = path.with_suffix('.prev.pt')
        backup = previous.with_suffix('.tmp')
        shutil.copyfile(path, backup)
        backup.replace(previous)
    temporary.replace(path)


def restore_epoch(path, model, optimizer, loader, scheduler=None):
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        previous = path.with_suffix('.prev.pt')
        if not previous.exists():
            return []
        path = previous
    try:
        saved = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        previous = path.with_suffix('.prev.pt')
        if not previous.exists():
            raise
        saved = torch.load(previous, map_location='cpu', weights_only=True)
        print('Using previous intact epoch checkpoint.', flush=True)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(saved['scheduler'])
    random.setstate(saved['python_rng'])
    state = saved['numpy_rng']
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if saved['loader_rng'] is not None:
        loader.generator.set_state(saved['loader_rng'])
    print(f'Restored {len(saved["history"])} completed epochs from {path.name}', flush=True)
    return saved['history']
