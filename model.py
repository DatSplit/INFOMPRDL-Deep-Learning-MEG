from time import time
import torch
import torch.nn as nn
from typing import List, Tuple
from torch.utils.data import DataLoader


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Training function
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> List[float]:
    start_time = time()
    print(f"Training Started:")

    model.to(device)
    batch_size = len(train_loader)
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        for i, (X, y) in enumerate(train_loader):
            X = X.float().to(device)
            y = y.long().to(device)

            preds = model(X)
            loss = loss_fn(preds, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch: [{epoch+1}/{num_epochs}], Batch: [{i}/{batch_size}], Loss: {loss.item():.4f}, Time: {time() - start_time:.2f}"
                )

    print(f"Total Training Time: {time() - start_time}")
    return train_losses


def test_model(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.float().to(device)
            y = y.long().to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    accuracy = accuracy * 100
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
