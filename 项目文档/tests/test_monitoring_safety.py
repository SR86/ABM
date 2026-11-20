import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from abm.monitoring import accuracy_probe
from abm.checkpoints import save_epoch, restore_epoch


def test_probe_preserves_rng_batchnorm_and_training_modes():
    torch.manual_seed(10)
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5), nn.Linear(3, 2)).train()
    model[-1].eval()
    data = TensorDataset(torch.randn(12, 3), torch.arange(12) % 2)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    modes = [m.training for m in model.modules()]
    accuracy_probe(model, data, 'cpu', 4)
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, before[key]) for key,value in model.state_dict().items())


def test_corrupted_epoch_uses_previous_backup(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.zeros(4, dtype=torch.long)),
                        batch_size=2, generator=torch.Generator().manual_seed(5))
    path = tmp_path / 'epoch.pt'
    save_epoch(path, model, optimizer, [dict(epoch=1)], loader)
    save_epoch(path, model, optimizer, [dict(epoch=1), dict(epoch=2)], loader)
    path.write_bytes(b'interrupted file')
    assert restore_epoch(path, model, optimizer, loader) == [dict(epoch=1)]
    path.unlink()
    assert restore_epoch(path, model, optimizer, loader) == [dict(epoch=1)]
