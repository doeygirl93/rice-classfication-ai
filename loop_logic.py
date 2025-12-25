### do logic for 1 train step, 1 test step and then final function that does the whole thing with multiple epoches
### copied from my other model with a couple of tweaks
import torch
from tqdm import tqdm


def one_train_step(model, dataloader, loss_fn, optimizer, device):
    ### Does just one epoch of training
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # Diddy blud clac loss

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # optimize w/ backpass
        optimizer.zero_grad()
        loss.backward()

        # taking step
        optimizer.step()

        # calc metrics

        y_pred_class = (torch.sigmoid(y_pred) > 0.5).float()
        train_acc += (y_pred_class == y.veiw_as(y_pred)).sum().item() / len(y_pred)

    return train_loss / len(dataloader), train_acc / len(dataloader)


def one_test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():  # lowk same as no grad but like more optimized n efficenint
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            batch_loss = loss_fn(test_pred, y)
            test_loss += batch_loss.item()

            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    return test_loss / len(dataloader), test_acc / len(dataloader)


def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device):
    """basically combines the two and make it multiple epoches"""

    # HOLY FREAK IS THIS SO SMART AND WONDERFUL
    results = {

        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = one_train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = one_test_step(model, test_dataloader, loss_fn, device)

        print(
            f"Epoch; {epoch + 1} --- Train loss: {train_loss:.4f} --- Test loss: {test_loss:.4f} --- Test accuracy: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
