import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import LSTMModel, train_model
from utlis import (
    CLASS_SIZE,
    CROSS_TRAIN_PATH,
    DEVICE,
    FEATURE_SIZE,
    INTRA_TRAIN_PATH,
    MEGDataset,
)


# Hyper parameter list
## Preprocessing
sequence_length = 10
downsample_size = 2

## Training
training_epochs = 1
batch_size = 1000
learning_rate = 0.0001
loss_function = nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam

## Model Features
hidden_size = 128
number_of_layers = 2
dropout = 0.2


def setup_data_and_train(training_folder_path: str):
    model = LSTMModel(
        input_size=FEATURE_SIZE,
        hidden_size=hidden_size,
        num_layers=number_of_layers,
        num_classes=CLASS_SIZE,
        dropout=0.2,
    )
    train_dataset = MEGDataset(
        sequence_length=sequence_length,
        downsample_size=downsample_size,
        data_path=training_folder_path,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optimizer_function(model.parameters(), lr=learning_rate)

    print("Training Model")
    train_model(
        model,
        train_dataloader,
        loss_function,
        optimizer,
        num_epochs=training_epochs,
        device=DEVICE,
    )
    return model


def select_training_model(dataset_type):
    if dataset_type == "cross":
        print("Preparing Cross Dataset")
        model = setup_data_and_train(CROSS_TRAIN_PATH)
        torch.save(model, "checkpoints/cross_model.pth")
    elif dataset_type == "intra":
        print("Preparing Intra Dataset")
        model = setup_data_and_train(INTRA_TRAIN_PATH)
        torch.save(model, "checkpoints/intra_model.pth")
    else:
        raise ValueError("Invalid dataset type")


def main():
    parser = argparse.ArgumentParser(description="Dataset type processing script.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cross", "intra"],
        required=True,
        help="Type of dataset to process: 'cross' or 'intra'",
        default="intra",
    )

    args = parser.parse_args()
    select_training_model(args.dataset)


if __name__ == "__main__":
    main()
