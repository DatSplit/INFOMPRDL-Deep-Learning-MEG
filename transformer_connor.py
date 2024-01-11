import argparse
import torch
import torch.nn as nn
from transformer.model import TransformerTimeSeriesClassifier
from transformer.test import test_model
from transformer.train import train_model
from utlis import CLASS_SIZE, DEVICE, FEATURE_SIZE, MEGDatasetType, get_dataloader


# Hyper parameter list
## Data
sequence_length = 2500
shuffle = True

## Training
training_epochs = 10
batch_size = 3
learning_rate = 0.000001
criterion = nn.NLLLoss()
optimizer_function = torch.optim.Adam
validation_split = 0.15

## Model Features
d_model = 256
n_head = 8
num_encoder_layers = 3
dim_feedforward = 1024
dropout = 0.1


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

    model = TransformerTimeSeriesClassifier(
        input_features=FEATURE_SIZE,
        num_classes=CLASS_SIZE,
        seq_len=sequence_length,
        d_model=d_model,
        n_head=n_head,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    optimizer = optimizer_function(model.parameters(), lr=learning_rate)

    if args.train_set:
        train_dataloader, eval_dataloader = get_dataloader(
            train_map[args.train_set],
            batch_size=batch_size,
            sequence_length=sequence_length,
            shuffle=shuffle,
            load_all_data=True,
            validation_split=validation_split,
        )
        print("Training Model")
        train_model(
            model,
            train_dataloader,
            eval_dataloader,
            criterion,
            optimizer,
            num_epochs=training_epochs,
            device=DEVICE,
        )

    if args.test_set:
        test_dataloader = get_dataloader(
            test_map[args.test_set],
            batch_size=batch_size,
            sequence_length=sequence_length,
            load_all_data=True,
        )

        model.load_state_dict(torch.load("checkpoints/transformer_model.pth"))

        test_model(test_dataloader, model, criterion, DEVICE)

    if not args.test_set and not args.train_set:
        print("No Options Selected!!")


if __name__ == "__main__":
    main()
