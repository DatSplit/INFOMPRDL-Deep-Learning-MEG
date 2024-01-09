import argparse
import itertools
from time import time
import torch
import torch.nn as nn
from typing import List
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import CLASS_SIZE, DEVICE, FEATURE_SIZE, MEGDatasetType, get_dataloader

# Data
sequence_length = 10
shuffle = True

# Training
num_epochs = 2
batch_size = 500
learning_rate = 0.0001
loss_function = nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam

number_of_layers = 2

# Define GRU Network
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch: [{epoch + 1}/{num_epochs}], Batch: [{i}/{batch_size}], Loss: {loss.item():.4f}, Time: {time() - start_time:.2f}"
                )

    print(f"Total Training Time: {time() - start_time}, Last Loss: {loss.item():.4f}")
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
    return test_loss, accuracy


def main():
    train_map = {
        "intra": MEGDatasetType.INTRA_TRAIN,
        "cross": MEGDatasetType.CROSS_TRAIN,
    }

    test_map = {
        "intra": MEGDatasetType.INTRA_TEST,
        "cross1": MEGDatasetType.CROSS_TEST_1,
        "cross2": MEGDatasetType.CROSS_TEST_2,
        "cross3": MEGDatasetType.CROSS_TEST_3,
    }

    # Hyper parameter list
    ## Data
    sequence_length = 10
    shuffle = True

    ## Training
    training_epochs = 2
    batch_size = 500
    learning_rate = 0.0001
    loss_function = nn.CrossEntropyLoss()
    optimizer_function = torch.optim.Adam

    ## Model Features
    hidden_size = 128
    number_of_layers = 2
    dropout = 0.2

    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument(
        "--test_set",
        type=str,
        choices=["intra", "cross1", "cross2", "cross3"],
        required=False,
        help="Type of dataset to test: 'intra', 'cross1', 'cross2' or 'cross3'",
        default=None,
    )

    parser.add_argument(
        "--train_set",
        type=str,
        choices=["cross", "intra"],
        required=False,
        help="Type of dataset to process: 'cross' or 'intra'",
        default=None,
    )

    parser.add_argument(
        "--hyperparameter_tuning",
        type=str,
        choices=["cross", "intra", "cross1", "cross2", "cross3"],
        required=False,
        help="Type of dataset to process: 'cross' or 'intra'",
        nargs="+",
        default=None,
    )
    args = parser.parse_args()
    model = GRU(
        input_size=FEATURE_SIZE,
        hidden_size=hidden_size,
        num_layers=number_of_layers,
        num_classes=CLASS_SIZE,
        #dropout=dropout,
    )
    optimizer = optimizer_function(model.parameters(), lr=learning_rate)

    if args.train_set:
        train_dataloader = get_dataloader(
            train_map[args.train_set],
            batch_size=batch_size,
            sequence_length=sequence_length,
            shuffle=shuffle,
            load_all_data=True,
        )
        print("Training Model")
        train_model(
            model,
            train_dataloader,
            loss_function,
            optimizer,
            num_epochs=training_epochs,
            device=DEVICE,
        )
        torch.save(model, f"checkpoints/{args.train_set}_model.pth")

    if args.test_set:
        test_dataloader = get_dataloader(
            test_map[args.test_set],
            batch_size=batch_size,
            sequence_length=sequence_length,
            load_all_data=True,
        )
        if "cross" in args.test_set:
            model = torch.load("checkpoints/cross_model.pth")
        else:
            model = torch.load("checkpoints/intra_model.pth")

        test_model(test_dataloader, model, loss_function, DEVICE)

    if args.hyperparameter_tuning:
        hyperparameters = {
            'learning_rates': [0.001, 0.01, 0.1],
            'batch_size': [128, 256, 500],
            'sequence_length': [10, 20, 40, 80, 160, 500],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
        }

        hyperparameter_combinations = list(itertools.product(*[hyperparameters[key] for key in hyperparameters]))
        print(len(hyperparameter_combinations), hyperparameter_combinations)

        for lr, batch_size, seq_length, drop in hyperparameter_combinations:
            model = GRU(
                input_size=FEATURE_SIZE,
                hidden_size=hidden_size,
                num_layers=number_of_layers,
                num_classes=CLASS_SIZE,
                dropout=drop,
            )
            optimizer = optimizer_function(model.parameters(), lr=lr)

            train_dataloader = get_dataloader(
                train_map[args.hyperparameter_tuning[0]],
                batch_size=batch_size,
                sequence_length=seq_length,
                shuffle=shuffle,
                load_all_data=True,
            )

            test_dataloader = get_dataloader(
                test_map[args.hyperparameter_tuning[1]],
                batch_size=batch_size,
                sequence_length=seq_length,
                shuffle=shuffle,
                load_all_data=True,
            )

            train_model(
                model,
                train_dataloader,
                loss_function,
                optimizer,
                num_epochs=training_epochs,
                device=DEVICE,
            )

            loss, accuracy = test_model(test_dataloader, model, loss_function, DEVICE)

            with open('hyperparameter_tuning_results.txt', 'a') as file:
                file.write(
                    f"Learning rate: {lr}, Batch size: {batch_size}, Sequence length: {seq_length}, Dropout: {drop}, Test loss: {loss}, Accuracy: {accuracy}\n")

            torch.save(model,
                       f"checkpoints/lr{lr}_batch_size{batch_size}_seq_length{seq_length}_dropout{drop}_model.pth")

        print("Hyperparameter tuning complete")

    if not args.test_set and not args.train_set and not args.hyperparameter_tuning:
        print("No Options Selected!!")


if __name__ == "__main__":
    main()
