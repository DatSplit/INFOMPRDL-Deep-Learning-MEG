import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import test_model
from utlis import (
    CROSS_TEST_1_PATH,
    CROSS_TEST_2_PATH,
    CROSS_TEST_3_PATH,
    DEVICE,
    INTRA_TEST_PATH,
    MEGDataset,
)

# Hyper parameter list
sequence_length = 10
downsample_size = 2
batch_size = 64
loss_function = nn.CrossEntropyLoss()


def select_test(test: str):
    if test == "intra":
        print(f"Testing {test}")

        model = torch.load("checkpoints/intra_model.pth")
        data_path = INTRA_TEST_PATH
    elif test == "cross1":
        print(f"Testing {test}")

        model = torch.load("checkpoints/cross_model.pth")
        data_path = CROSS_TEST_1_PATH
    elif test == "cross2":
        print(f"Testing {test}")

        model = torch.load("checkpoints/cross_model.pth")
        data_path = CROSS_TEST_2_PATH
    elif test == "cross3":
        print(f"Testing {test}")

        model = torch.load("checkpoints/cross_model.pth")
        data_path = CROSS_TEST_3_PATH

    else:
        raise ValueError("Invalid dataset type")

    test_dataset = MEGDataset(
        sequence_length=sequence_length,
        downsample_size=1,
        data_path=data_path,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    test_model(test_dataloader, model, loss_function, DEVICE)


def main():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument(
        "--testset",
        type=str,
        choices=["intra", "cross1", "cross2", "cross3"],
        required=True,
        help="Type of dataset to test: 'intra', 'cross1', 'cross2' or 'cross3'",
        default="intra",
    )

    args = parser.parse_args()
    select_test(args.testset)


if __name__ == "__main__":
    main()
