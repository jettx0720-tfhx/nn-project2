"""
VGG_Residual: VGG-style network with residual connections for CIFAR-10.
~4.8M parameters. Each stage has two Conv-BN-ReLU blocks with skip connection.
"""

import torch
import torch.nn as nn
from utils.nn import init_weights_


class ResidualBlock(nn.Module):
    """Two Conv-BN-ReLU blocks with a skip connection."""

    def __init__(self, in_channels, out_channels, stride=1, use_1x1=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.use_1x1 = use_1x1
        if use_1x1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class VGG_Residual(nn.Module):
    """
    VGG-inspired residual network with 4 stages.
    Stage config: 64-128-256-256 channels.
    Uses AdaptiveAvgPool + single Linear for classification.
    ~4.8M parameters.
    """

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(inp_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 channels, 32x32 -> 16x16
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, use_1x1=False),
            ResidualBlock(64, 64, stride=1, use_1x1=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2: 128 channels, 16x16 -> 8x8
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1, use_1x1=True),
            ResidualBlock(128, 128, stride=1, use_1x1=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 3: 256 channels, 8x8 -> 4x4
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=1, use_1x1=True),
            ResidualBlock(256, 256, stride=1, use_1x1=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 4: 256 channels, 4x4
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 256, stride=1, use_1x1=False),
            ResidualBlock(256, 256, stride=1, use_1x1=False),
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        self.apply(init_weights_)


if __name__ == '__main__':
    from .vgg import get_number_of_parameters
    model = VGG_Residual()
    print(f"VGG_Residual: {get_number_of_parameters(model):,} parameters")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
