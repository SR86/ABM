import torch
import pytest
from author_compat.resnet import resnet


@pytest.mark.parametrize('classes', [10, 11])
def test_author_four_output_interface_and_suppression_gradient(classes):
    torch.set_num_threads(1)
    model = resnet(depth=20, num_classes=classes)
    image = torch.rand(2, 3, 32, 32)
    triggered = image.clone()
    triggered[:, :, :2, :2] = 1
    first, second, near, logits = model(image)
    assert first.shape == (2, 16, 32, 32)
    assert second.shape == (2, 32, 16, 16)
    assert near.shape == (2, 64, 8, 8)
    assert logits.shape == (2, classes)
    near2 = model(triggered)[2]
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2, dtype=torch.long)) + (near-near2).square().sum()/2000
    loss.backward()
    assert torch.isfinite(loss)
    assert model.conv1.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('mode, shape', [('pooled', (2, 64)), ('logits', (2, 10))])
def test_near_modes_explicitly_mark_reconstruction_hypotheses(mode, shape):
    torch.set_num_threads(1)
    model = resnet(depth=20, num_classes=10, near_mode=mode).eval()
    assert model(torch.zeros(2, 3, 32, 32))[2].shape == shape
