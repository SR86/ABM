"""Residual networks; CIFAR/GTSRB use a 3x3 stem without max pooling."""
import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=10, small_input=True):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3 if small_input else 7,
                               1 if small_input else 2, 1 if small_input else 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity() if small_input else nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, blocks[0], 1)
        self.layer2 = self._make_layer(128, blocks[1], 2)
        self.layer3 = self._make_layer(256, blocks[2], 2)
        self.layer4 = self._make_layer(512, blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, channels, count, stride):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(count - 1))
        return nn.Sequential(*layers)

    def forward_with_features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        features = torch.flatten(self.avgpool(x), 1)
        return self.fc(features), features

    def forward(self, x):
        return self.forward_with_features(x)[0]


class ResNet18(ResNet):
    def __init__(self, num_classes=10):
        super().__init__([2, 2, 2, 2], num_classes, small_input=True)


def build_model(dataset, num_classes):
    if dataset == 'imagenet_subset':
        return ResNet([3, 4, 6, 3], num_classes, small_input=False)
    return ResNet18(num_classes)
