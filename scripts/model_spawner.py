import torch
from torch import nn


class VGG(nn.Module):
    def __init__(self, input, output, hidden_units) -> None:
        super().__init__()
        # self.input = input
        # self.output = output
        # self.hidden_units = hidden_units

        self.conv_block = nn.Sequential(
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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=12 * 12 * hidden_units,
                out_features=output,  # Equals number of classes
            ),
        )

    def forward(self, x) -> torch.Tensor:
        return self.classifier(self.conv_block(x))
