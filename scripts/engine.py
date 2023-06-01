import torch


def train_step(model, dataloader, loss_fn, device, optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
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

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            acc = torch.eq(y_pred_class, y).sum().item() / len(y)
            test_acc += acc

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc


def train(model, train_dataloader, test_dataloader, loss_fn, device, optimizer, epochs):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, device, optimizer
        )

        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch + 1}|Test Acc: {test_acc:.2f}|Train Acc: {train_acc:.2f}|Train Loss: {train_loss:.3f}|Test Loss: {test_loss:.3f}|"
        )

    return results
