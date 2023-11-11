import torch
import torchvision
from torch import nn
from torchvision.models import efficientnet_b0

# This script is used for configuration of pretrained models.


class EfficientNetMNIST(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Get pretrained weights
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = torchvision.models.efficientnet_b0(weights="DEFAULT")

        # Freeze the "feature extractor"
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False

        # Adjust EfficientNet for MNIST dataset
        in_features = 1280
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, 10)
        )

    def forward(self, x):
        return self.efficientnet(x)


# ! Czy effnet nie będzie miał zbyt wielkiej warstwy wejściowej? XD


if __name__ == "__main__":
    pass
