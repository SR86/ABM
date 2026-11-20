"""Candidate for ABM's missing CIFAR ResNet; NOT the recovered original file.

Architecture/interface matches NAD's public CIFAR BasicBlock ResNet reference.
This module is isolated from abm.models and does not alter running experiments.
"""
import math
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if inputs != outputs or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False),
                                            nn.BatchNorm2d(outputs))

    def forward(self, image):
        identity = image if self.downsample is None else self.downsample(image)
        branch = self.relu(self.bn1(self.conv1(image)))
        return self.relu(self.bn2(self.conv2(branch)) + identity)


class CifarResNet(nn.Module):
    def __init__(self, depth=20, num_classes=10, block_name='BasicBlock', near_mode='stage3'):
        super().__init__()
        if block_name.lower() != 'basicblock' or depth < 8 or (depth - 2) % 6:
            raise ValueError('Candidate supports BasicBlock CIFAR depths 6n+2 only')
        if near_mode not in ('stage3', 'pooled', 'logits'):
            raise ValueError('near_mode must be stage3, pooled, or logits')
        self.near_mode = near_mode
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        count = (depth - 2) // 6
        channels = 16
        for index, width in enumerate((16, 32, 64), 1):
            stride = 1 if index == 1 else 2
            blocks = [ResidualBlock(channels, width, stride)]
            blocks.extend(ResidualBlock(width, width, 1) for _ in range(count - 1))
            setattr(self, f'layer{index}', nn.Sequential(*blocks))
            channels = width
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0, math.sqrt(2 / (layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels)))
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 32, 32):
            raise ValueError('Candidate expects NCHW RGB 32x32 CIFAR inputs')
        first = self.layer1(self.relu(self.bn1(self.conv1(image))))
        second = self.layer2(first)
        third = self.layer3(second)
        pooled = self.avgpool(third).flatten(1)
        logits = self.fc(pooled)
        near = {'stage3': third, 'pooled': pooled, 'logits': logits}[self.near_mode]
        return first, second, near, logits


def resnet(**kwargs):
    return CifarResNet(**kwargs)
