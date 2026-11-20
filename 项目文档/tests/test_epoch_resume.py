import random
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from abm.method import train_teacher, train_student, NBAB
from abm.data import TensorImageDataset


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def model(classes):
    result = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 12), nn.Dropout(.2), nn.Linear(12, classes))
    if classes == 3:
        with torch.no_grad():
            result[-1].bias[2] = -10
    return result


@pytest.mark.parametrize('stage', ['teacher', 'student', 'nbab'])
def test_epoch_resume_matches_uninterrupted_training_exactly(tmp_path, stage):
    torch.set_num_threads(1)
    data = TensorImageDataset(torch.linspace(0, 1, 8 * 3 * 8 * 8).reshape(8, 3, 8, 8), torch.arange(8) % 2)

    def run(epochs, checkpoint=None):
        student = model(3 if stage == 'nbab' else 2)
        loader = DataLoader(data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(17))
        if stage == 'teacher':
            history = train_teacher(student, loader, 'cpu', epochs=epochs, lr=.01, checkpoint_path=checkpoint)
        elif stage == 'student':
            teacher = model(2)
            history = train_student(teacher, student, loader, 'cpu', epochs=epochs, checkpoint_path=checkpoint)
        else:
            nbab = NBAB(num_classes=2, trigger_size=1, alpha=.1, control_rate=.25, isolation_rate=.5, batch_size=4)
            nbab.create_isolated_dataset(data, student, epochs=epochs, checkpoint_path=checkpoint)
            history = nbab.history
        return {k:v.clone() for k,v in student.state_dict().items()}, history, (random.random(), float(np.random.rand()), torch.rand(2))

    seed(23)
    expected, history, rng = run(4)
    seed(23)
    path = tmp_path / 'epoch.pt'
    run(2, path)
    seed(987)  # Resumed process starts from unrelated RNG/model initialization.
    # Student's teacher represents an already saved frozen stage.
    if stage == 'student':
        seed(23)
    actual, resumed_history, resumed_rng = run(4, path)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert history == resumed_history
    assert rng[:2] == resumed_rng[:2]
    torch.testing.assert_close(rng[2], resumed_rng[2], rtol=0, atol=0)
    assert path.with_suffix('.prev.pt').exists()
