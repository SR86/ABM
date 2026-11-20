"""Numerical and behavioral checks for the paper's ABM pipeline.

These tests use synthetic tensors to verify the equations and information
boundaries. They do not establish the accuracy reported by the paper.
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from abm.attacks import BadNetsAttack, BlendAttack
from abm.data import AttackTestDataset, TensorImageDataset, load_datasets
from abm.method import (
    ABM,
    NBAB,
    abm_logits,
    compute_average_activation,
    nbab_loss,
    train_student,
    train_teacher,
)
from abm.models import BasicBlock, ResNet18


@pytest.fixture(autouse=True)
def reproducible_tensors():
    torch.manual_seed(7)
    torch.set_num_threads(1)


class SmallClassifier(nn.Module):
    """Includes BN buffers so an accidentally trained teacher is detectable."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        if x.ndim == 4:
            x = x.mean(dim=(-1, -2))
        return self.fc(self.bn(x))


def test_residual_block_retains_identity_when_residual_path_is_zero():
    block = BasicBlock(3, 3).eval()
    for layer in block.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    # Positive input survives the post-addition ReLU without any change.
    x = torch.rand(2, 3, 8, 8)
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)


def test_residual_projection_and_cifar_classifier_shapes():
    block = BasicBlock(3, 8, stride=2).eval()
    assert block(torch.rand(2, 3, 16, 16)).shape == (2, 8, 8, 8)
    model = ResNet18(num_classes=10).eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 32, 32)).shape == (2, 10)


def test_nbab_loss_matches_ce_and_only_triggered_pairs_are_regularized():
    logits = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, -1.0, 0.0], [1.0, 0.0, -2.0]],
        requires_grad=True,
    )
    original = torch.tensor(
        [[30.0, -20.0, 10.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([1, 2, 2])
    mask = torch.tensor([False, True, True])
    alpha = 3.0
    loss = nbab_loss(logits, labels, original, mask, alpha)
    expected = F.cross_entropy(logits, labels) + alpha * mask.float().mean() * F.mse_loss(
        logits[mask], original[mask]
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None and original.grad is not None
    torch.testing.assert_close(original.grad[0], torch.zeros(3))
    assert original.grad[mask].abs().sum() > 0


def test_nbab_without_control_samples_is_plain_classification():
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    original = torch.full((4, 3), 100.0, requires_grad=True)
    loss = nbab_loss(logits, labels, original, torch.zeros(4, dtype=torch.bool), 1000.0)
    torch.testing.assert_close(loss, F.cross_entropy(logits, labels))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_nbab_explicit_control_subset_reduction_is_distinct_from_batch_mean():
    logits = torch.tensor([[1., 0.], [0., 1.]])
    original = torch.zeros_like(logits)
    labels, mask = torch.tensor([0, 1]), torch.tensor([True, False])
    ce = F.cross_entropy(logits, labels)
    batch = nbab_loss(logits, labels, original, mask, 2., 'batch')
    triggered = nbab_loss(logits, labels, original, mask, 2., 'triggered')
    torch.testing.assert_close(triggered - ce, 2 * (batch - ce))


def test_equation_four_operates_directly_on_class_logits():
    teacher = torch.tensor([[3.0, 5.0, 7.0], [-2.0, 0.0, 9.0]])
    student = torch.tensor([[0.0, 1.0, 3.0], [-1.0, 4.0, 2.0]])
    output = abm_logits(teacher, student, torch.tensor(2.0), delta=2.0, gamma=0.5)
    expected = torch.tensor([[3.0, 5.0, 3.0], [-2.0, -6.0, 7.0]])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.shape == teacher.shape


def test_equation_four_supports_a_documented_per_class_threshold():
    teacher = torch.tensor([[5.0, 5.0, 5.0]])
    student = torch.tensor([[2.0, 2.0, 2.0]])
    output = abm_logits(teacher, student, torch.tensor([0.0, 1.0, 3.0]), delta=1.0, gamma=1.0)
    torch.testing.assert_close(output, torch.tensor([[3.0, 4.0, 5.0]]))


def test_activation_uses_label_logits_and_counts_the_incomplete_last_batch():
    logits = torch.tensor(
        [[2.0, 4.0, 6.0], [10.0, 12.0, 14.0], [20.0, 22.0, 24.0]]
    )
    labels = torch.tensor([0, 2, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    avg = compute_average_activation(nn.Identity(), loader, torch.device("cpu"), mode="label")
    torch.testing.assert_close(torch.as_tensor(avg), torch.tensor((2.0 + 14.0 + 22.0) / 3))
    per_class = compute_average_activation(nn.Identity(), loader, torch.device("cpu"), mode="class")
    torch.testing.assert_close(torch.as_tensor(per_class), logits.mean(dim=0))


def test_student_training_never_changes_teacher_weights_or_bn_buffers():
    teacher, student = SmallClassifier(), SmallClassifier()
    teacher.train()
    before_teacher = {key: value.clone() for key, value in teacher.state_dict().items()}
    before_student = {key: value.clone() for key, value in student.state_dict().items()}
    data = TensorDataset(torch.rand(8, 3, 4, 4), torch.arange(8) % 3)
    loader = DataLoader(data, batch_size=4, shuffle=False)
    train_student(teacher, student, loader, torch.device("cpu"), epochs=2, lr=0.02)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before_teacher[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not teacher.training
    assert any(
        not torch.equal(value, before_student[key])
        for key, value in student.state_dict().items()
        if value.dtype.is_floating_point
    )


def test_abm_wrapper_does_not_put_teacher_in_training_mode():
    teacher, student = SmallClassifier(), SmallClassifier()
    defended = ABM(teacher, student, torch.tensor(2.0))
    defended.train()
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    defended.eval()
    images = torch.rand(2, 3, 4, 4)
    with torch.no_grad():
        expected = abm_logits(teacher(images), student(images), torch.tensor(2.0))
        torch.testing.assert_close(defended(images), expected)


def test_badnets_uses_attack_location_and_does_not_mutate_input():
    image = torch.zeros(3, 32, 32)
    attack = BadNetsAttack()
    output = attack.add_trigger(image)
    expected = torch.zeros_like(image)
    expected[:, 27:29, 27:29] = 1.0
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.count_nonzero(image) == 0
    assert torch.count_nonzero(output[:, :2, :2]) == 0


def test_poisoning_is_reproducible_and_does_not_change_source_labels():
    images = torch.zeros(20, 3, 32, 32)
    labels = torch.arange(20) % 10
    data = TensorImageDataset(images, labels)
    attack = BadNetsAttack(target_label=0, poison_rate=0.25, seed=11)
    poisoned, first_indices = attack.poison_dataset(data)
    _, second_indices = BadNetsAttack(target_label=0, poison_rate=0.25, seed=11).poison_dataset(data)
    assert list(first_indices) == list(second_indices)
    assert len(first_indices) == 5
    selected = set(int(index) for index in first_indices)
    for index in range(len(data)):
        image, label = poisoned[index]
        if index in selected:
            assert int(label) == 0
            assert torch.all(image[:, 27:29, 27:29] == 1)
        else:
            assert int(label) == int(labels[index])
            assert torch.count_nonzero(image) == 0
    assert torch.count_nonzero(images) == 0
    torch.testing.assert_close(labels, torch.arange(20) % 10)


@pytest.mark.parametrize("mode", ["fixed", "samplewise"])
def test_blend_trigger_is_reproducible_in_pixel_space(mode):
    image = torch.full((3, 32, 32), 0.5)
    original = image.clone()
    one = BlendAttack(alpha=0.2, image_size=32, seed=11, pattern_mode=mode)
    two = BlendAttack(alpha=0.2, image_size=32, seed=11, pattern_mode=mode)
    output = one.add_trigger(image, index=3)
    torch.testing.assert_close(output, two.add_trigger(image, index=3))
    assert output.min() >= 0.4 and output.max() <= 0.6
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    if mode == "samplewise":
        assert not torch.equal(output, one.add_trigger(image, index=4))


def test_attack_test_dataset_excludes_target_class_from_asr_denominator():
    images = torch.zeros(6, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 2, 1, 0])
    data = TensorImageDataset(images, labels)
    triggered = AttackTestDataset(data, BadNetsAttack(target_label=0))
    assert len(triggered) == 3
    for image, target in triggered:
        assert int(target) == 0
        assert torch.all(image[:, 27:29, 27:29] == 1)


class FilterProbeDataset(torch.utils.data.Dataset):
    """Poison annotations raise if a defense attempts to consult them."""

    def __init__(self):
        self.images = torch.tensor([0.0, 0.25, 0.5, 0.75]).view(4, 1, 1, 1).expand(4, 3, 4, 4).clone()

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.images[index], index % 2

    @property
    def poison_indices(self):
        raise AssertionError("NBAB must not use ground-truth poison annotations")


class FixedVerificationModel(nn.Module):
    """Predicts from a pixel untouched by the controllable 2x2 trigger."""

    def __init__(self, predict_new_only=False):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.predict_new_only = predict_new_only

    def forward(self, x):
        strength = x[:, 0, 2, 2]
        new_class = torch.ones_like(strength) if self.predict_new_only else (strength < 0.5).float()
        return torch.stack([1 - new_class, torch.zeros_like(strength), new_class], dim=1) + self.dummy * 0


def test_nbab_filters_by_predictions_without_poison_oracle_and_returns_originals():
    dataset = FilterProbeDataset()
    nbab = NBAB(num_classes=2, isolation_rate=0.5, batch_size=2, seed=7)
    isolated = nbab.create_isolated_dataset(dataset, FixedVerificationModel(), epochs=0)
    assert set(nbab.candidate_indices.tolist()) == {2, 3}
    assert set(isolated.indices) == {2, 3}
    for source_index, (image, label) in zip(isolated.indices, isolated):
        original, original_label = dataset[source_index]
        torch.testing.assert_close(image, original, rtol=0, atol=0)
        assert label == original_label
        assert torch.all(image[:, :2, :2] < 1)


def test_nbab_refuses_empty_candidates_instead_of_using_oracle_fallback():
    nbab = NBAB(num_classes=2, isolation_rate=0.5, batch_size=2)
    with pytest.raises(ValueError, match="zero candidates"):
        nbab.create_isolated_dataset(FilterProbeDataset(), FixedVerificationModel(True), epochs=0)


def test_teacher_schedule_uses_the_papers_explicit_two_epoch_decay():
    model = SmallClassifier()
    loader = DataLoader(TensorDataset(torch.rand(8, 3, 4, 4), torch.arange(8) % 3), batch_size=4)
    history = train_teacher(model, loader, "cpu", epochs=5, lr=0.1, step_size=2, lr_gamma=0.1)
    assert [record["lr"] for record in history] == pytest.approx([0.1, 0.1, 0.01, 0.01, 0.001])


def _external_payload():
    return {
        "train_images": torch.zeros(4, 3, 32, 32, dtype=torch.uint8),
        "train_labels": torch.tensor([0, 1, 2, 0]),
        "clean_test_images": torch.zeros(3, 3, 32, 32, dtype=torch.uint8),
        "clean_test_labels": torch.tensor([0, 1, 2]),
        "attack_test_images": torch.zeros(3, 3, 32, 32, dtype=torch.uint8),
        "attack_test_labels": torch.zeros(3, dtype=torch.long),
    }


def test_external_asr_requires_auditable_non_target_semantics(tmp_path):
    path = tmp_path / "attack.pt"
    torch.save(_external_payload(), path)
    config = dict(dataset="external", external_data=str(path), num_classes=3, target_label=0)
    with pytest.raises(ValueError, match="attack_test_original_labels"):
        load_datasets(config)


def test_external_asr_excludes_original_target_and_keeps_already_poisoned_inputs(tmp_path):
    path = tmp_path / "attack.pt"
    payload = _external_payload()
    payload["train_images"][:, :, 10, 10] = 255
    payload["attack_test_original_labels"] = torch.tensor([0, 1, 2])
    payload["poison_indices"] = torch.tensor([1, 3])
    torch.save(payload, path)
    train, clean, attack = load_datasets(
        dict(dataset="external", external_data=str(path), num_classes=3, target_label=0)
    )
    assert len(train) == 4 and len(clean) == 3 and len(attack) == 2
    assert train.poison_indices == [1, 3]
    image, label = train[1]
    assert label == 1
    assert torch.all(image[:, 10, 10] == 1)
    assert torch.count_nonzero(image[:, :2, :2]) == 0
    assert attack.original_labels == [1, 2]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01, 1.01])
def test_tensor_dataset_rejects_invalid_pixel_space(invalid):
    images = torch.zeros(1, 3, 4, 4)
    images[0, 0, 0, 0] = invalid
    with pytest.raises(ValueError):
        TensorImageDataset(images, torch.tensor([0]))

