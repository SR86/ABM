"""Measure whether separate BN forwards penalize unchanged images as well."""
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from author_compat.resnet import resnet

torch.set_num_threads(2)
torch.manual_seed(42)
raw = CIFAR10(str(ROOT/'data/cifar10'), train=True, download=False, transform=ToTensor())
original = torch.stack([raw[i][0] for i in range(120)])
triggered = original.clone()
mask = torch.arange(120) < 12
triggered[mask, :, :2, :2] = 1
base = resnet(depth=20, num_classes=11, near_mode='stage3')
records = {}
with torch.no_grad():
    for mode in ('joint', 'separate'):
        model = copy.deepcopy(base).train()
        if mode == 'joint':
            a,b = model(torch.cat([triggered, original]))[2].chunk(2)
        else:
            a,b = model(triggered)[2], model(original)[2]
        diff = (a-b).square()
        records[mode] = {'control_squared_sum':diff[mask].sum().item(),
                         'unchanged_squared_sum':diff[~mask].sum().item(),
                         'all_sum_div_2000':diff.sum().item()/2000,
                         'bn_batches':int(model.bn1.num_batches_tracked)}
output = ROOT/'results/nbab_forward_audit/pairing_bn_probe.json'
output.write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
