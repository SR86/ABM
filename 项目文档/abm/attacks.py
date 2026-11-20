"""BadNets and Blend triggers checked against the released ABM data loader.

Reference: data_loader_Bab.py, _BadNetTrigger (27:29), _BlendTrigger (random
grayscale uint8 mask, alpha=0.2, then uint8 quantization). No downloaded code
is executed. All methods accept unnormalized CHW floating tensors in [0, 1].
"""

import math
import numbers

import numpy as np
import torch

from .data import PoisonedDataset


def _validate_image(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise ValueError("add_trigger expects a CHW torch tensor")
    if not image.is_floating_point() or min(image.shape) <= 0:
        raise ValueError("add_trigger expects a nonempty floating tensor in [0, 1]")
    if not torch.isfinite(image).all().item() or image.min().item() < 0 or image.max().item() > 1:
        raise ValueError("Add triggers before Normalize: image values must be finite and in [0, 1]")


class BackdoorAttack:
    def __init__(self, target_label=0, poison_rate=0.1, seed=42):
        if isinstance(target_label, bool) or not isinstance(target_label, numbers.Integral) or target_label < 0:
            raise ValueError("target_label must be a nonnegative integer")
        if not isinstance(poison_rate, numbers.Real) or not math.isfinite(poison_rate) or not 0 <= poison_rate <= 1:
            raise ValueError("poison_rate must be in [0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.target_label = int(target_label)
        self.poison_rate = float(poison_rate)
        self.seed = int(seed)

    def add_trigger(self, image, index=None):
        raise NotImplementedError

    def poison_dataset(self, dataset):
        """Poison floor(rate*N) uniformly sampled training items, including targets.

        Sampling from all classes follows the authors' all-to-one implementation.
        Original dataset tensors and global NumPy random state remain unchanged.
        """
        if len(dataset) == 0:
            raise ValueError("Cannot poison an empty dataset")
        count = int(len(dataset) * self.poison_rate)
        indices = sorted(np.random.RandomState(self.seed).permutation(len(dataset))[:count].tolist())
        return PoisonedDataset(dataset, self, indices), indices


class BadNetsAttack(BackdoorAttack):
    """A white 2x2 square at rows/columns 27:29 for 32x32 inputs.

    The separate NBAB controllable trigger uses position=(0, 0); it must not be
    confused with this attacker trigger from the released implementation.
    """

    def __init__(self, target_label=0, poison_rate=0.1, trigger_size=2, position=(27, 27), seed=42):
        super().__init__(target_label, poison_rate, seed)
        if isinstance(trigger_size, bool) or not isinstance(trigger_size, numbers.Integral) or trigger_size < 1:
            raise ValueError("trigger_size must be a positive integer")
        if not isinstance(position, (tuple, list)) or len(position) != 2 or any(
            isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0 for value in position
        ):
            raise ValueError("position must contain nonnegative integer (row, column)")
        self.trigger_size = int(trigger_size)
        self.position = tuple(int(value) for value in position)

    def add_trigger(self, img, index=None):
        _validate_image(img)
        row, column = self.position
        size = self.trigger_size
        if row + size > img.shape[1] or column + size > img.shape[2]:
            raise ValueError(f"Trigger at {self.position} with size {size} does not fit image {tuple(img.shape)}")
        result = img.clone()
        result[:, row:row + size, column:column + size] = 1.0
        return result


class BlendAttack(BackdoorAttack):
    """Blend a grayscale uint8 random pattern into the image, then quantize.

    'fixed' uses one seeded pattern for all images. 'samplewise' follows the
    released code's per-image randomness, with a stable seed/index mapping so
    repeated epochs and worker counts cannot change the attack. The test wrapper
    assigns a separate index namespace. Neither mode reproduces the authors'
    unrecorded global RNG history bit-for-bit.
    """

    def __init__(self, target_label=0, poison_rate=0.1, alpha=0.2, image_size=32, seed=42, pattern_mode="fixed"):
        super().__init__(target_label, poison_rate, seed)
        if not isinstance(alpha, numbers.Real) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Blend alpha must be in [0, 1]")
        if isinstance(image_size, bool) or not isinstance(image_size, numbers.Integral) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if pattern_mode not in {"fixed", "samplewise"}:
            raise ValueError("Blend pattern_mode must be 'fixed' or 'samplewise'")
        self.alpha = float(alpha)
        self.image_size = int(image_size)
        self.pattern_mode = pattern_mode
        self.trigger = self._make_pattern(None)

    def _make_pattern(self, index):
        if self.pattern_mode == "fixed":
            rng = np.random.RandomState(self.seed)
        else:
            index = 0 if index is None else index
            if isinstance(index, bool) or not isinstance(index, numbers.Integral) or index < 0:
                raise ValueError("Blend sample index must be a nonnegative integer")
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        mask = rng.integers(0, 256, (self.image_size, self.image_size), dtype=np.uint8) if isinstance(rng, np.random.Generator) else rng.randint(0, 256, (self.image_size, self.image_size), dtype=np.uint8)
        return torch.from_numpy(mask.copy()).unsqueeze(0)

    def add_trigger(self, img, index=None):
        _validate_image(img)
        if tuple(img.shape[1:]) != (self.image_size, self.image_size):
            raise ValueError(f"Blend expects {self.image_size}x{self.image_size} images, got {tuple(img.shape[1:])}")
        if self.alpha == 0:
            return img.clone()
        pattern = self.trigger if self.pattern_mode == "fixed" else self._make_pattern(index)
        # The released implementation blends uint8 PIL pixels and truncates the
        # result to uint8. Round ToTensor values back to their original pixels
        # first, avoiding float32 conversion errors at integer boundaries.
        pixels = img.to(torch.float64).mul(255).round()
        pattern = pattern.to(device=img.device, dtype=torch.float64)
        result = ((1 - self.alpha) * pixels + self.alpha * pattern).floor().clamp(0, 255)
        return result.div(255).to(img.dtype)
