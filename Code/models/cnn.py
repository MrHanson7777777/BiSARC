#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CNN model definitions —— MNIST / FEMNIST / CIFAR-10 / CIFAR-100"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    """Generic Residual Block"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CNNMnist(nn.Module):
    """Small CNN for MNIST (28x28 grayscale images)"""

    def __init__(self, num_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)                     # 28 -> 14
        x = F.relu(self.conv2(x))
        x = self.pool(x)                     # 14 -> 7
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x                             

class CNNCifar(nn.Module):
    """CNN with residual blocks for CIFAR-10 / CIFAR-100 (32x32 color images)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 32×32
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 16×16
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 8×8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride):
        layers = [_ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(_ResBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class CNNFemnist(nn.Module):
    """
    CNN for FEMNIST (28x28 grayscale images, 62 classes).

    Architecture refers to the LEAF benchmark and McMahan et al. (2017) "Communication-Efficient
    Learning of Deep Networks from Decentralized Data",
    but appropriately extended for 62 classes.

    Architecture:
        Conv2d(1, 32, 5, padding=2) → ReLU → MaxPool(2)    # 28→14
        Conv2d(32, 64, 5, padding=2) → ReLU → MaxPool(2)   # 14→7
        FC(64*7*7, 512) → ReLU → Dropout(0.5)
        FC(512, num_classes)
    """

    def __init__(self, num_channels=1, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)                     # 28 -> 14
        x = F.relu(self.conv2(x))
        x = self.pool(x)                     # 14 -> 7
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)            # flatten: (B, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x