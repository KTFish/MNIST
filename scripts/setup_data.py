from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision


def create_datasets(train_dir, test_dir, transform=torchvision.transforms.ToTensor()):
    # Create MNIST Train and Test Datasets
    train_dataset = MNIST(
        root=train_dir,
        train=True,
        download=True,
        transform=transform,
        target_transform=None,
    )

    test_dataset = MNIST(
        root=test_dir,
        train=False,
        download=True,
        transform=transform,
        target_transform=None,
    )

    return train_dataset, test_dataset, train_dataset.classes


def create_dataloaders(train_dir, test_dir, batch_size=32):
    # Create datasets and get class names
    train_dataset, test_dataset, class_names = create_datasets(train_dir, test_dir)

    # Turn datasets into dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, class_names
