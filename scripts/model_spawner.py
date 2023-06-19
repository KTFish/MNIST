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


class VggWandb(nn.Module):
    def __init__(
        self,
        input: int,
        output: int,
        hidden_units: int,
        kernel_size: int,
        padding: int,
        stride: int,
    ) -> None:
        """VGG Architecture connected with `Weights and Biases` library for experiment tracking.

        Args:
            input (int): Number of neurons in the input layer.
            output (int): Number of neurons in the output layer.
            hidden_units (int): Number of neurons in dense layer.
        """
        super().__init__()
        # n --> Shape of the feature map after traversing through all convolutional layers
        conv = lambda n: (n - kernel_size + 2 * padding) / stride + 1
        pool = lambda n: (n - 2) / 2 + 1
        layer = lambda n: pool(conv(n))
        image_input_shape = 28
        self.n = int(layer(layer(image_input_shape)))

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input,  # RBG,
                out_channels=hidden_units,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.n * self.n * hidden_units,
                out_features=output,  # Equals number of classes
            ),
        )

    def forward(self, x) -> torch.Tensor:
        # print("n", self.n)
        # print("input", x.shape)
        # x = self.conv_block_1(x)
        # print("conv1", x.shape)
        # x = self.conv_block_2(x)
        # print("conv2", x.shape)
        # x = self.classifier(x)
        # print("output", x.shape)
        # return x
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
