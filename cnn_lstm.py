import argparse
from time import time
import torch
import torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import CLASS_SIZE, DEVICE, FEATURE_SIZE, MEGDatasetType, get_dataloader
import itertools

#Set device to GPU or CPU based on availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


    input_folder_path = "../input/"

# Hyper parameter list
## Data
sequence_length = 1
shuffle = True

## Training
training_epochs = 2
batch_size = 500
learning_rate = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam

## Model Features
hidden_size = 128
number_of_layers = 2
dropout = 0.2
kernel_size = 6
stride = 1
padding = 3
pool_size = 2
pool_stride = 2

class CNN_LSTM_Model(nn.Module):
    def __init__(
            self, 
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            kernel_size: int,
            stride: int,
            padding: int,
            pool_size: int,
            pool_stride: int,
            dropout: float
            ):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), x.size(2), -1)
        out, _ = self.lstm(x)
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

    scheduler = ReduceLROnPlateau(optimizer, 'min')
    model.to(device)
    batch_size = len(train_loader)
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        for i, (X, y) in enumerate(train_loader):
            X = X.float().to(device)  
            #print(X.size())
            #X = X.unsqueeze(3)

            # X = X.unsqueeze(1) # adds dimension add index 1
            # Need to add dimension add index 3?
           
            #X = X.reshape(500, 10, 248, 248)  
            #print(X.size())
            # X = X.permute(0, 2, 3, 1)
            #print(X.size())
            y = y.long().to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step(loss)

            if i % 100 == 0:
                print(
                    f"Epoch: [{epoch+1}/{num_epochs}], Batch: [{i}/{batch_size}], Loss: {loss.item():.4f}, Time: {time() - start_time:.2f}"
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
            # X = X.unsqueeze(1)
            # X = X.permute(0, 2, 3, 1)
            #X = X.unsqueeze(3)
            y = y.long().to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    accuracy = accuracy * 100
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")


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

    # Hyper parameter list
    ## Data
    sequence_length = 1
    shuffle = True

    ## Training
    training_epochs = 2
    batch_size = 500
    learning_rate = 0.001
    loss_function = nn.CrossEntropyLoss()
    optimizer_function = torch.optim.Adam

    ## Model Features
    hidden_size = 128
    number_of_layers = 2
    dropout = 0.2
    kernel_size = 6
    stride = 1
    padding = 3
    pool_size = 2
    pool_stride = 2

    model = CNN_LSTM_Model(
        input_size=FEATURE_SIZE,
        hidden_size=hidden_size,
        num_layers=number_of_layers,
        num_classes=CLASS_SIZE,
        kernel_size = kernel_size,
        stride = stride,
        padding=padding,
        pool_size=pool_size,
        pool_stride=pool_stride,
        dropout=dropout,
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
            'sequence_length': [1, 10, 20, 40, 80, 160, 500],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            #'hidden_size': [64,128,256]
        }

        hyperparameter_combinations = list(itertools.product(*[hyperparameters[key] for key in hyperparameters]))
        print(len(hyperparameter_combinations), hyperparameter_combinations)

        #keep these variables for CNN standard:
        stride = 1
        pool_size = 2
        pool_stride = 2

        for lr, batch_size, seq_length, drop in hyperparameter_combinations:

            #first for kernel_size=3 and padding=2
            kernel_size = 3
            padding = 2

            model = CNN_LSTM_Model(
                input_size=FEATURE_SIZE,
                hidden_size=hidden_size,
                num_layers=number_of_layers,
                num_classes=CLASS_SIZE,
                kernel_size = kernel_size,
                stride = stride,
                padding=padding,
                pool_size=pool_size,
                pool_stride=pool_stride,
                dropout=dropout,
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
                file.write(f"Learning rate: {lr}, Batch size: {batch_size}, Sequence length: {seq_length}, Dropout: {drop}, Kernel size: {kernel_size}, Padding: {padding}, Test loss: {loss}, Accuracy: {accuracy}\n")

            torch.save(model, f"checkpoints/lr{lr}_batch_size{batch_size}_seq_length{seq_length}_dropout{drop}_kernel_size{kernel_size}_padding{padding}_model.pth")


            # now for kernel_size=6 and padding=3
            kernel_size = 6
            padding = 3

            model = CNN_LSTM_Model(
                input_size=FEATURE_SIZE,
                hidden_size=hidden_size,
                num_layers=number_of_layers,
                num_classes=CLASS_SIZE,
                kernel_size = kernel_size,
                stride = stride,
                padding=padding,
                pool_size=pool_size,
                pool_stride=pool_stride,
                dropout=dropout,
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
                file.write(f"Learning rate: {lr}, Batch size: {batch_size}, Sequence length: {seq_length}, Dropout: {drop}, Kernel size: {kernel_size}, Padding: {padding}, Test loss: {loss}, Accuracy: {accuracy}\n")

            torch.save(model, f"checkpoints/lr{lr}_batch_size{batch_size}_seq_length{seq_length}_dropout{drop}_kernel_size{kernel_size}_padding{padding}_model.pth")


            # now for kernel_size=9 and padding=5
            kernel_size = 9
            padding = 5

            model = CNN_LSTM_Model(
                input_size=FEATURE_SIZE,
                hidden_size=hidden_size,
                num_layers=number_of_layers,
                num_classes=CLASS_SIZE,
                kernel_size = kernel_size,
                stride = stride,
                padding=padding,
                pool_size=pool_size,
                pool_stride=pool_stride,
                dropout=dropout,
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
                file.write(f"Learning rate: {lr}, Batch size: {batch_size}, Sequence length: {seq_length}, Dropout: {drop}, Kernel size: {kernel_size}, Padding: {padding}, Test loss: {loss}, Accuracy: {accuracy}\n")

            torch.save(model, f"checkpoints/lr{lr}_batch_size{batch_size}_seq_length{seq_length}_dropout{drop}_kernel_size{kernel_size}_padding{padding}_model.pth")

        print("Hyperparameter tuning complete")

    if not args.test_set and not args.train_set:
        print("No Options Selected!!")


if __name__ == "__main__":
    main()
