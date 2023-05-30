import torch
import os
import matplotlib.pyplot as plt
from scripts import setup_data

# Paths
train_dir = r"C:\Users\janek\notebooks\0_Projects\KAGGLE\data\mnist\train"
test_dir = r"C:\Users\janek\notebooks\0_Projects\KAGGLE\data\mnist\test"

# Create folders if needed
if not os.path.exists(train_dir):
    os.makedirs(train_dir, exist_ok=True)

if not os.path.exists(test_dir):
    os.makedirs(test_dir, exist_ok=True)

BATCH_SIZE = 32

# Create dataloaders using `setup_data.py` script
train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, batch_size=BATCH_SIZE
)

### Visualise an image
image_batch, label_batch = next(iter(train_dataloader))
# Get single image from batch
image, label = image_batch[0], label_batch[0]
print(image.shape, label.shape)

# Plot
plt.imshow(image.permute(1, 2, 0), cmap="gray")
plt.title(class_names[label])
plt.show()

### Setup a base line model
from scripts import model_spawner

assert len(class_names) == 10  # Because it is MNIST dataset

model = model_spawner.VGG(input=1, output=len(class_names), hidden_units=5)
optimizer = torch.optim.SGD(lr=0.01, params=model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()  # Because wue have multiple classes

# setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

### Training & Testing Loop
epochs = 1
for epoch in range(epochs):
    # Train Step
    train_loss, train_acc = 0, 0
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)

        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        y_pred_class = torch.argmax(y_logits, dim=1)
        acc = torch.eq(y_pred_class, y).sum().item() / len(y)
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= BATCH_SIZE
    train_acc /= BATCH_SIZE

    # Test Step
    test_loss, test_acc = 0, 0
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)

        loss = loss_fn(y_logits, y)
        test_loss += loss.item()

        y_pred_class = torch.argmax(y_logits, dim=1)
        acc = torch.eq(y_pred_class, y).sum().item() / len(y)
        test_acc += acc

    test_loss /= BATCH_SIZE
    test_acc /= BATCH_SIZE

print(f"Epoch {epoch + 1} | Train Loss {train_loss:.3f} Train Acc: {train_acc:.3f}")

# We've done!
