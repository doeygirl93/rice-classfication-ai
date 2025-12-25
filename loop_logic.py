import torch
from tqdm import tqdm

def one_train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y.unsqueeze(1))
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        y_pred_class = (torch.sigmoid(y_pred) > 0.5).float()            #TS HEAVILY DEPENDS ON THE DATATYPE. THIS ONE IS SIMPLY FOR BINARY DATA THATS TABULAR DON'T USE WITH IMAGES
        train_acc += (y_pred_class == y.view_as(y_pred)).sum().item()

    return train_loss / len(dataloader), train_acc / len(dataloader)

def one_test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            batch_loss = loss_fn(test_pred, y.unsqueeze(1))
            test_loss += batch_loss.item()

            test_pred_labels = (torch.sigmoid(test_pred) > 0.5).float()
            test_acc += (test_pred_labels == y.view_as(test_pred)).sum().item() / len(test_pred)

    return test_loss / len(dataloader), test_acc / len(dataloader)




def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device):  #THIS COMBINES EVERYTHING AND MAKES IT RUN FOR MUTIPLE EPOCHES
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
