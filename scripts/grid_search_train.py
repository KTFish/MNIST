import wandb
import torch
import setup_data
import model_spawner


def train(config=None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with wandb.init(config):
        # Get hyperparameters from config dict
        config = wandb.config
        BATCH_SIZE = config.batch_size
        HIDDEN_UNITS = config.hidden_units
        EPOCHS = config.epochs
        KERNEL_SIZE = config.kernel_size
        PADDING = config.padding
        STRIDE = config.stride
        LEARNING_RATE = config.lr
        print(config)

        # Setup dataloaders using `setup_data.py` script
        train_dataloader, test_datalaoder, class_names = setup_data.create_dataloaders(
            batch_size=BATCH_SIZE
        )

        # Setup model using hyperparameters from config
        # model = model_spawner.VGG(input=1, output=10, hidden_units=HIDDEN_UNITS).to(
        #     device
        # )

        model = model_spawner.VggWandb(
            input=1,  # Gray Scale Image (Not RBG)
            output=10,
            hidden_units=HIDDEN_UNITS,
            kernel_size=KERNEL_SIZE,
            padding=PADDING,
            stride=STRIDE,
        ).to(device)

        print("[INFO] Model initialized correctly...")

        # Setup Loss and Optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())

        # Train the model
        for epoch in range(EPOCHS):
            # Train Step
            train_loss, train_acc = 0, 0
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)

                y_logits = model(X)
                # Calculate Loss
                loss = loss_fn(y_logits, y)
                train_loss += loss.item()

                # Calculate accuracy
                y_pred_classes = torch.argmax(y_logits, dim=1)
                train_acc += torch.eq(y_pred_classes, y).sum().item() / len(y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)

            # Test step
            test_loss, test_acc = 0, 0
            for X, y in test_datalaoder:
                X, y = X.to(device), y.to(device)

                y_logits = model(X)
                # Calculate Loss
                loss = loss_fn(y_logits, y)
                test_loss += loss.item()

                # Calculate accuracy
                y_pred_classes = torch.argmax(y_logits, dim=1)
                test_acc += torch.eq(y_pred_classes, y).sum().item() / len(y)

            test_loss /= len(test_datalaoder)
            test_acc /= len(test_datalaoder)

            # Print out in console
            print(
                f"Epoch [{epoch}] || Train Accuracy: {train_acc:.2f} || Test Accuracy: {test_acc:.2f} || Train Loss: {train_loss:.3f} || Test Loss: {test_loss:.3f}"
            )

            # Log results
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
