import argparse
from time import time
import torch
import torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import CLASS_SIZE, DEVICE, FEATURE_SIZE, MEGDatasetType, get_dataloader

#Set device to GPU or CPU based on availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


    input_folder_path = "../input/"

# Hyper parameter list
## Data
sequence_length = 10
shuffle = True

## Training
training_epochs = 10
batch_size = 500
learning_rate = 0.003
loss_function = nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam

## Model Features
#hidden_size = 128
#number_of_layers = 2
dropout = 0.2
kernel_size = 8
stride = 1
padding = 4
pool_size = 2
pool_stride = 2

class CNNModel(nn.Module):
    def __init__(
            self, 
            kernel_size: int,
            stride: int,
            padding: int,
            pool_size: int,
            pool_stride: int,
            dropout: float
            #add layer sizes
            ):
        super(CNNModel, self).__init__()
        
        #First convolution layer     
        self.conv_1 = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride))

        #Second convolution layer       
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 60, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride))
        
        #add dropout? 
        #Dropout
        self.dropout = nn.Dropout(p=dropout)
     

        #Fully connected layers
        self.fc1 = nn.Linear(7*7*32, 128)       
        self.fc2 = nn.Linear(128, 10)        
        
    def forward(self, x):  
        # Reshape the tensor 
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = out.view(out.size(0), -1)
        #dropout?
        out = self.dropout(out)
        # Calculate the flattened size
        flattened_size = out.size(1)

        # Adjust the input size of first linear layer to match the flattened size
        self.fc1 = nn.Linear(flattened_size, 128)

        out = self.fc1(out)

        # Apply an activation function (if needed)
        # out = torch.relu(out)

        out = self.fc2(out)

        out=self.dropout(out)
        # out = self.fc1(out)
        # out = self.fc2(out)        
        out = F.log_softmax(out,dim=1)                                
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
            # X = X.unsqueeze(1) # adds dimension add index 1
            # Need to add dimension add index 3?
            X = X.unsqueeze(3)
            #X = X.reshape(500, 10, 248, 248)  
            #print(X.size())
            # X = X.permute(0, 2, 3, 1)
            # print(X.size())
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
            X = X.unsqueeze(3)
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
    args = parser.parse_args()
    model = CNNModel(
        kernel_size = kernel_size,
        stride = stride,
        padding=padding,
        pool_size=pool_size,
        pool_stride=pool_stride,
        dropout=dropout
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

    if not args.test_set and not args.train_set:
        print("No Options Selected!!")


if __name__ == "__main__":
    main()
