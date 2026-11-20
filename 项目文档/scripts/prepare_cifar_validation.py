"""Prepare small real-CIFAR validation exports from verified archive members.

The archive can still be downloading: only the complete data_batch_4 and
official test_batch members are consumed, each with torchvision's expected MD5.
The source archive is read only; arbitrary tar paths are never extracted.

Run from the project root:
    python scripts/prepare_cifar_validation.py
"""

import argparse
import hashlib
import io
import json
import pickle
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abm.attacks import BadNetsAttack, BlendAttack
from abm.data import AttackTestDataset, TensorImageDataset
from abm.runner import validate_config


EXPECTED = {
    "cifar-10-batches-py/data_batch_4": "634d18415352ddfa80567beed471001a",
    "cifar-10-batches-py/test_batch": "40351d587109b95175f43aff81a1287e",
}


def project_path(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def verified_members(source, destination):
    """Read exact regular members; validate length+MD5 before writing or use."""
    members = {}
    try:
        with tarfile.open(source, "r|gz") as archive:
            for member in archive:
                if member.name not in EXPECTED:
                    continue
                if not member.isfile():
                    raise ValueError(f"Not a regular archive member: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                raw = stream.read()
                digest = hashlib.md5(raw).hexdigest()
                if len(raw) != member.size or digest != EXPECTED[member.name]:
                    raise ValueError(f"Incomplete/corrupt {member.name}: length={len(raw)}/{member.size}, MD5={digest}")
                target = destination / Path(member.name).name
                if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != digest:
                    temporary = target.with_suffix(".tmp")
                    temporary.write_bytes(raw)
                    temporary.replace(target)
                members[member.name] = raw
                print(f"Verified {member.name}: MD5 {digest}", flush=True)
                if len(members) == len(EXPECTED):
                    break
    except (EOFError, tarfile.ReadError, OSError) as exc:
        raise ValueError(f"Archive prefix has not reached both complete members; retry after more downloading: {exc}") from exc
    if set(members) != set(EXPECTED):
        raise ValueError("Both complete official train/test members are required.")
    return members


def sample_batch(raw, count, seed):
    # Unpickle only AFTER a complete member matches torchvision's fixed MD5.
    batch = pickle.load(io.BytesIO(raw), encoding="bytes")
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    if count < 10 or count > len(labels):
        raise ValueError(f"Sample count must be in [10, {len(labels)}]")
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(10):
        required = count // 10 + int(class_id < count % 10)
        available = np.flatnonzero(labels == class_id)
        if required > len(available):
            raise ValueError(f"Not enough examples of class {class_id} for stratified sampling")
        selected.extend(rng.permutation(available)[:required])
    indices = np.asarray(selected, dtype=np.int64)
    rng.shuffle(indices)
    return (torch.from_numpy(images[indices].copy()), torch.from_numpy(labels[indices].copy()),
            torch.from_numpy(indices.copy()))


def save_payload_if_changed(path, payload):
    if path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(previous, dict) and set(previous) == set(payload) and all(
            torch.equal(previous[key], value) if isinstance(value, torch.Tensor)
            else previous[key] == value for key, value in payload.items()
        ):
            print(f"Reused existing identical export: {project_path(path)}", flush=True)
            return
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/cifar10/cifar-10-python.tar.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cifar10_validation")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = args.archive.resolve()
    initial_size = source.stat().st_size
    members = verified_members(source, destination)
    train_images, train_labels, train_indices = sample_batch(members["cifar-10-batches-py/data_batch_4"], args.train_samples, 42)
    test_images, test_labels, test_indices = sample_batch(members["cifar-10-batches-py/test_batch"], args.test_samples, 43)
    train_data, test_data = TensorImageDataset(train_images, train_labels), TensorImageDataset(test_images, test_labels)
    manifest = {
        "source_archive": project_path(source),
        "source_bytes_at_start": initial_size,
        "whole_archive_md5_verified": False,
        "official_torchvision_archive_md5": "c58f30108f718f92721af3b95e74349a",
        "verification_scope": "Only the two complete named members were MD5 verified; no claim of complete-archive verification.",
        "member_md5": {Path(name).name: digest for name, digest in EXPECTED.items()},
        "torchvision_reference": "https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py",
        "train_member": "data_batch_4",
        "test_member": "test_batch",
        "official_test_split": True,
        "train_samples": args.train_samples,
        "clean_test_samples": args.test_samples,
        "sampling": "Stratified without replacement; numpy default_rng seeds 42 (train), 43 (official test); remainder goes to lowest class IDs.",
        "validation_only": True,
        "note": "Small real-data pipeline validation. This is not a reproduction of the full paper metrics.",
        "payloads": {},
    }
    for attack_name, constructor in (("badnets", BadNetsAttack), ("blend", BlendAttack)):
        attack = constructor(target_label=0, poison_rate=0.1, seed=42)
        poisoned, poison_indices = attack.poison_dataset(train_data)
        asr = AttackTestDataset(test_data, attack)
        payload = {
            "train_images": torch.stack([poisoned[i][0] for i in range(len(poisoned))]).mul(255).round().to(torch.uint8),
            "train_labels": torch.tensor([poisoned[i][1] for i in range(len(poisoned))]),
            "clean_test_images": test_images,
            "clean_test_labels": test_labels,
            "attack_test_images": torch.stack([asr[i][0] for i in range(len(asr))]).mul(255).round().to(torch.uint8),
            "attack_test_labels": torch.zeros(len(asr), dtype=torch.long),
            "attack_test_original_labels": test_labels[asr.indices],
            "poison_indices": torch.tensor(poison_indices, dtype=torch.long),
            "train_source_indices": train_indices,
            "clean_test_source_indices": test_indices,
            "attack_test_source_indices": test_indices[asr.indices],
            "metadata": {"dataset": "cifar10", "attack": attack_name, "train_member": "data_batch_4", "test_member": "test_batch", "seed": 42, "real_data_validation_only": True},
        }
        payload_path = destination / f"{attack_name}_{args.train_samples}_{args.test_samples}.pt"
        save_payload_if_changed(payload_path, payload)
        config = json.loads((ROOT / f"configs/cifar10_{attack_name}.json").read_text(encoding="utf-8"))
        config.update(dataset="external", attack="external", model_dataset="cifar10", external_data=project_path(payload_path),
                      nbab_epochs=10, nbab_lr=0.01, teacher_epochs=2, teacher_lr=0.001, student_epochs=2, student_lr=0.001,
                      max_train_samples=args.train_samples, max_test_samples=args.test_samples, batch_size=64,
                      output_dir=f"results/cifar10_{attack_name}_validation")
        validate_config(config)
        config_path = ROOT / f"configs/validation_{attack_name}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["payloads"][attack_name] = {
            "path": project_path(payload_path), "config": project_path(config_path),
            "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            "poisoned_train_count": len(poison_indices), "asr_test_samples": len(asr),
            "blend_pattern_mode": "fixed" if attack_name == "blend" else None,
        }
        print(json.dumps(manifest["payloads"][attack_name]), flush=True)
    (destination / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Real CIFAR-10 validation exports and configurations are ready.", flush=True)


if __name__ == "__main__":
    main()
