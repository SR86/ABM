"""Lazy image datasets in the authors' unnormalized [0, 1] input space.

No random crop, flip, normalization, or other augmentation is applied. External
attack datasets are already poisoned: loading them never inserts a trigger.
"""

from pathlib import Path
import numbers
from typing import Any

import torch
from torch.utils.data import Dataset, Subset


_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _positive_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("image_size must be a positive integer")
    return value


def _indices(values, length: int, name: str = "indices") -> list[int]:
    tensor = torch.as_tensor(values) if not isinstance(values, torch.Tensor) else values
    if tensor.numel() == 0 and tensor.ndim == 1:
        return []
    if tensor.ndim != 1 or tensor.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    result = tensor.detach().cpu().tolist()
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate entries")
    if min(result) < 0 or max(result) >= length:
        raise ValueError(f"{name} must be in [0, {length})")
    return result


def dataset_targets(dataset: Dataset) -> list[int]:
    """Get labels without decoding images when the underlying dataset exposes them."""
    if isinstance(dataset, Subset):
        parent_targets = dataset_targets(dataset.dataset)
        return [int(parent_targets[index]) for index in dataset.indices]
    if hasattr(dataset, "targets"):
        values = dataset.targets
    elif hasattr(dataset, "labels"):
        values = dataset.labels
    elif hasattr(dataset, "samples"):
        values = [sample[1] for sample in dataset.samples]
    else:
        values = [dataset[index][1] for index in range(len(dataset))]
    result = [int(value) for value in values]
    if len(result) != len(dataset):
        raise ValueError("dataset label count does not match dataset length")
    return result


class TensorImageDataset(Dataset):
    """An NCHW RGB tensor dataset; uint8 storage is converted lazily to float32.

    Floating point images must already be finite and in [0, 1]. Labels must
    have an integer dtype. NHWC arrays and normalized images are rejected.
    """

    def __init__(self, images, labels, *, num_classes=None, image_size=None, name="dataset"):
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError(f"{name}: images and labels must be torch tensors")
        if images.device.type != "cpu" or labels.device.type != "cpu":
            raise ValueError(f"{name}: tensors must be on CPU")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"{name}: images must have shape [N, 3, H, W] (NCHW RGB)")
        if images.shape[0] == 0 or min(images.shape[2:]) == 0:
            raise ValueError(f"{name}: the image dataset must not be empty")
        if labels.ndim != 1 or len(labels) != len(images):
            raise ValueError(f"{name}: labels must have shape [N] matching images")
        if labels.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name}: labels must have an integer dtype")
        if images.dtype != torch.uint8 and not images.is_floating_point():
            raise ValueError(f"{name}: images must be uint8 or floating point in [0, 1]")
        if image_size is not None:
            size = _positive_size(image_size)
            if tuple(images.shape[2:]) != (size, size):
                raise ValueError(f"{name}: expected {size}x{size} images, got {tuple(images.shape[2:])}")
        if images.is_floating_point():
            # Keep validation memory bounded for ImageNet-sized tensor exports.
            for chunk in images.split(128):
                if not torch.isfinite(chunk).all().item():
                    raise ValueError(f"{name}: images contain NaN or infinite values")
                if chunk.min().item() < 0 or chunk.max().item() > 1:
                    raise ValueError(f"{name}: floating images must be in [0, 1]; do not Normalize")
        if labels.min().item() < 0:
            raise ValueError(f"{name}: labels must be nonnegative")
        if num_classes is not None:
            if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError("num_classes must be an integer >= 2")
            if labels.max().item() >= num_classes:
                raise ValueError(f"{name}: labels must be in [0, {num_classes})")
        self.images = images
        self.labels = labels.to(torch.int64)
        self.targets = self.labels.tolist()
        # Only metadata for diagnostics. The defense does not receive these labels.
        self.poison_indices = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if image.dtype == torch.uint8:
            image = image.to(torch.float32).div(255)
        else:
            image = image.to(torch.float32)
        return image, int(self.labels[index])


class PoisonedDataset(Dataset):
    """Replace selected training items lazily, preserving the original dataset."""

    def __init__(self, dataset, attack, indices):
        self.dataset = dataset
        self.attack = attack
        self.poison_indices = _indices(indices, len(dataset), "poison_indices")
        self.indices = self.poison_indices
        self._poison_set = frozenset(self.poison_indices)
        self.targets = dataset_targets(dataset)
        for index in self.poison_indices:
            self.targets[index] = int(attack.target_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Canonicalize negative indices before checking membership.
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        image, label = self.dataset[index]
        if index in self._poison_set:
            return self.attack.add_trigger(image, index=index), int(self.attack.target_label)
        return image, int(label)


class AttackTestDataset(Dataset):
    """All-to-one ASR set, excluding original target-class examples.

    A separate index namespace makes samplewise Blend masks independent of the
    training split, even when a test item has the same integer index.
    """

    INDEX_OFFSET = 1 << 40

    def __init__(self, dataset, attack):
        self.dataset = dataset
        self.attack = attack
        self.indices = [i for i, label in enumerate(dataset_targets(dataset))
                        if label != attack.target_label]
        if not self.indices:
            raise ValueError("ASR test set is empty after excluding the target class")
        self.targets = [int(attack.target_label)] * len(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        image, _ = self.dataset[source_index]
        return self.attack.add_trigger(image, index=self.INDEX_OFFSET + source_index), int(self.attack.target_label)


def _load_external(config):
    path_value = config.get("external_data")
    if not path_value:
        raise ValueError("dataset='external' requires external_data pointing to a .pt tensor dictionary")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"External tensor dataset not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot safely load external dataset {path}; use a weights_only-compatible tensor dictionary: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("External dataset must be a dictionary of named tensors")
    required = ("train_images", "train_labels", "clean_test_images", "clean_test_labels",
                "attack_test_images", "attack_test_labels")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"External dataset is missing required keys: {', '.join(missing)}")
    kwargs = {"num_classes": config.get("num_classes", 10),
              "image_size": config.get("image_size", 32)}
    train = TensorImageDataset(payload["train_images"], payload["train_labels"], name="train", **kwargs)
    clean = TensorImageDataset(payload["clean_test_images"], payload["clean_test_labels"], name="clean_test", **kwargs)
    attack = TensorImageDataset(payload["attack_test_images"], payload["attack_test_labels"], name="attack_test", **kwargs)
    target = config.get("target_label", 0)
    if isinstance(target, bool) or not isinstance(target, numbers.Integral) or not 0 <= target < kwargs["num_classes"]:
        raise ValueError("target_label must be an integer in [0, num_classes)")
    if not torch.all(attack.labels == target).item():
        raise ValueError("attack_test_labels must all equal target_label for all-to-one ASR")
    if "attack_test_original_labels" in payload:
        original = payload["attack_test_original_labels"]
        if (not isinstance(original, torch.Tensor) or original.ndim != 1
                or original.dtype not in _INTEGER_DTYPES or len(original) != len(attack)):
            raise ValueError("attack_test_original_labels must be an integer tensor of shape [N] matching attack_test_images")
        if original.min().item() < 0 or original.max().item() >= kwargs["num_classes"]:
            raise ValueError("attack_test_original_labels must be in [0, num_classes)")
        keep = torch.nonzero(original != target, as_tuple=False).flatten().tolist()
        if not keep:
            raise ValueError("External ASR test set is empty after excluding original target-class examples")
        attack = Subset(attack, keep)
        attack.targets = [int(target)] * len(keep)
        attack.original_labels = original[keep].tolist()
    elif payload.get("attack_test_non_target_only") is not True:
        raise ValueError("External ASR requires attack_test_original_labels for target exclusion, or the explicit declaration attack_test_non_target_only=True")
    if "poison_indices" in payload:
        train.poison_indices = _indices(payload["poison_indices"], len(train), "poison_indices")
    return train, clean, attack


def load_datasets(config):
    """Return (train, clean_test, external_attack_test_or_None).

    Built-in datasets are clean here; the runner calls attack.poison_dataset.
    GTSRB and the 12-class ImageNet subset use ImageFolder directories at
    data_root/train/<class>/ and data_root/test/<class>/ with identical mappings.
    External train tensors must already contain the desired poisoning. Attack
    labels must all equal target_label. Supply attack_test_original_labels to
    exclude original target examples, or declare attack_test_non_target_only=True
    to certify that this exclusion was already done during export.
    """
    name = str(config.get("dataset", "cifar10")).lower()
    if name == "external":
        return _load_external(config)
    if name not in {"cifar10", "gtsrb", "imagenet_subset"}:
        raise ValueError(f"Unsupported dataset {name!r}; choose cifar10, gtsrb, imagenet_subset, or external")
    size = _positive_size(config.get("image_size", 224 if name == "imagenet_subset" else 32))
    root = Path(config.get("data_root", "data")).expanduser()
    try:
        from torchvision import datasets, transforms
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Dataset loading requires a working torchvision installation matching torch") from exc
    steps = []
    if name != "cifar10" or size != 32:
        steps.append(transforms.Resize((size, size)))
    steps.append(transforms.ToTensor())
    transform = transforms.Compose(steps)
    if name == "cifar10":
        if config.get("num_classes", 10) != 10:
            raise ValueError("cifar10 requires num_classes=10")
        try:
            train = datasets.CIFAR10(root=str(root), train=True, download=bool(config.get("download", False)), transform=transform)
            test = datasets.CIFAR10(root=str(root), train=False, download=bool(config.get("download", False)), transform=transform)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"Cannot load CIFAR-10 at {root}. Place cifar-10-batches-py there or enable download (a valid local archive can be extracted): {exc}") from exc
        return train, test, None
    expected_classes = 43 if name == "gtsrb" else 12
    if config.get("num_classes", expected_classes) != expected_classes:
        raise ValueError(f"{name} requires num_classes={expected_classes}")
    for split in ("train", "test"):
        if not (root / split).is_dir():
            raise FileNotFoundError(f"{name} requires ImageFolder layout: {root / split}/<class>/<image>")
    try:
        train = datasets.ImageFolder(str(root / "train"), transform=transform)
        test = datasets.ImageFolder(str(root / "test"), transform=transform)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Cannot read {name} ImageFolder dataset at {root}: {exc}") from exc
    if len(train.classes) != expected_classes or len(test.classes) != expected_classes:
        raise ValueError(f"{name} requires exactly {expected_classes} classes in both splits; got train={len(train.classes)}, test={len(test.classes)}")
    if train.class_to_idx != test.class_to_idx:
        raise ValueError(f"{name}: train/test class-to-index mappings differ")
    return train, test, None
