import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type='groupnorm', num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = self._make_norm_layer(planes, norm_type, num_groups)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = self._make_norm_layer(planes, norm_type, num_groups)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                self._make_norm_layer(planes, norm_type, num_groups),
            )

    def _make_norm_layer(self, channels, norm_type, num_groups):
        if norm_type == 'batchnorm':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'groupnorm':
            return nn.GroupNorm(min(num_groups, channels), channels)
        else:
            raise ValueError(f"Unsupported normalization: {norm_type}")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18Fed(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, norm_type='groupnorm', num_groups=8):
        super().__init__()
        self.in_planes = 64
        self.norm_type = norm_type
        self.num_groups = num_groups
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = self._make_norm_layer(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_norm_layer(self, channels):
        if self.norm_type == 'batchnorm':
            return nn.BatchNorm2d(channels)
        elif self.norm_type == 'groupnorm':
            return nn.GroupNorm(min(self.num_groups, channels), channels)
        else:
            raise ValueError(f"Unsupported normalization: {self.norm_type}")

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_type, self.num_groups))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet18(num_classes=10, image_size=32, in_channels=3):

    return ResNet18Fed(num_classes=num_classes, in_channels=in_channels, norm_type='groupnorm')

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type='groupnorm', num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = self._make_norm_layer(planes, norm_type, num_groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self._make_norm_layer(planes, norm_type, num_groups)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = self._make_norm_layer(self.expansion * planes, norm_type, num_groups)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                self._make_norm_layer(self.expansion * planes, norm_type, num_groups),
            )

    def _make_norm_layer(self, channels, norm_type, num_groups):
        if norm_type == 'batchnorm':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'groupnorm':
            return nn.GroupNorm(min(num_groups, channels), channels)
        else:
            raise ValueError(f"Unsupported normalization: {norm_type}")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

ResNet101 = None
ResNet50 = None