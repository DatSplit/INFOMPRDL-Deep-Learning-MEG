import argparse
from time import time
import torch
import torch.nn as nn
from typing import List
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import CLASS_SIZE, DEVICE, FEATURE_SIZE, MEGDatasetType, get_dataloader

# Two layers with 128 do not perform significantly better than with 64 (accuracy = 0.96) for crosstrain cross1test
# Better than 32 (acc = 0.92)
# performance of 64 improved to 0.989 with l2 regularization parameter = 0.01
# dropout does not have a noticeable effect, slightly decreases performance

# Data
sequence_length = 1
shuffle = True

# Training
num_epochs = 2
batch_size = 500
learning_rate = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam


# Define Neural Network
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):#, dropout_rate=0.5):
        super(FFNN, self).__init__()
        # Hidden Layers
        self.hidden1 = nn.Linear(input_size, hidden_size)
        #self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        #self.dropout2 = nn.Dropout(dropout_rate)
        # Activation function
        self.relu = nn.ReLU()
        # Output layer
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.relu(out)
        #out = self.dropout1(out)
        out = self.hidden2(out)
        out = self.relu(out)
        #out = self.dropout2(out)
        out = self.output(out)
        return out


# Training function
def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Adam,
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
            #print(X.shape)
            #print(y.shape, preds.shape)
            preds = preds.squeeze(1)
            loss = loss_fn(preds, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
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
            pred = pred.squeeze(1)
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
    args = parser.parse_args()
    model = FFNN(
        input_size=FEATURE_SIZE,
        hidden_size=64,
        num_classes=CLASS_SIZE,
    )
    optimizer = optimizer_function(model.parameters(), lr=learning_rate, weight_decay=0.01)

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
            num_epochs=num_epochs,
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

    if not args.test_set and not args.train_set:
        print("No Options Selected!!")


if __name__ == "__main__":
    main()

    '''
    dataloader = get_dataloader(MEGDatasetType.INTRA_TRAIN, batch_size=32, shuffle=True, sequence_length=10)
    count = 0
    for batch in dataloader:
        if count == 0:
            print(len(batch[0][0]), len(batch[1]))
        else:
            break
        count = 1
    '''