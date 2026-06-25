"""
SimpleCNN: A lightweight CNN baseline for CIFAR-10.
3 Conv + 3 MaxPool + 2 Linear layers.
~0.45M parameters -- about 1/20th of VGG_A.
"""

import torch.nn as nn
from utils.nn import init_weights_


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for quick experiments and parameter-efficiency baseline.
    3 conv stages with BN + Dropout in classifier.
    ~0.45M parameters.
    """

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(inp_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        self.apply(init_weights_)


if __name__ == '__main__':
    from .vgg import get_number_of_parameters
    model = SimpleCNN()
    print(f"SimpleCNN: {get_number_of_parameters(model):,} parameters")
