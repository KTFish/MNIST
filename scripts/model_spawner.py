import torch
from torch import nn


class VGG(nn.Module):
    def __init__(self, input, output, hidden_units) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input,  # RBG,
                out_channels=hidden_units,
                kernel_size=5,
                padding=0,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=0,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=5 * 5 * hidden_units,
                out_features=output,  # Equals number of classes
            ),
        )

    def forward(self, x) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


class BaseLine(nn.Module):
    def __init__(self, input, output, hidden_units) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output),
        )

    def forward(self, x):
        return self.classifier(x)
