import torch
import os
import matplotlib.pyplot as plt
from scripts import setup_data, model_spawner, engine, utils
from torchvision import transforms

# Setup Hyperparameters
BATCH_SIZE = 32
EPOCHOS = 10
HIDDEN_UNITS = 10

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup transforms
transforms = transforms.Compose(
    [
        transforms.Resize(28, 28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create dataloaders using `setup_data.py` script
train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders(
    batch_size=BATCH_SIZE, transform=transforms
)


# Plot a random sample from the dataset
utils.visualize_sample_image(class_names=class_names, dataloader=train_dataloader)

assert len(class_names) == 10  # MNIST has 10 classes (Digits 0 - 9)


### Setup a Model


# model = model_spawner.VGG(
#     input=1, output=len(class_names), hidden_units=HIDDEN_UNITS
# ).to(device)
# model = model_spawner.BaseLine(
#     input=28 * 28, output=len(class_names), hidden_units=HIDDEN_UNITS
# ).to(device)
optimizer = torch.optim.SGD(lr=0.01, params=model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()  # Because wue have multiple classes

### Training & Testing Loop
results = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHOS,
    device=device,
)
