import argparse
import torch
import torch.nn as nn
from transformer.model import TransformerTimeSeriesClassifier
from transformer.test import test_model
from transformer.train import train_model
from utils import CLASS_SIZE, DEVICE, FEATURE_SIZE, MEGDatasetType, get_dataloader
from torch.utils.data import DataLoader, ConcatDataset, Subset
import random
from ray import train, tune

# Hyper parameter list
## Data
sequence_length = 100
shuffle = True

## Training
training_epochs = 1
batch_size = 2
learning_rate = 0.00001
criterion = nn.NLLLoss()
optimizer_function = torch.optim.Adam
validation_split = 0.15

## Model Features
d_model = 512
n_head = 8
num_encoder_layers = 3
dim_feedforward = 1024
dropout = 0.2


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
        "--tune_cross",
        type=bool,
        choices=[True, False],
        required=False,
        default=False,
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
    if args.tune_cross:

        def train_mnist(config):
            train_dataloader, eval_dataloader = get_dataloader(
                train_map["cross"],
                batch_size=config["batch_size"],
                sequence_length=config["sequence_length"],
                shuffle=shuffle,
                load_all_data=True,
                validation_split=0.05,
            )
            cross1_dataloader = get_dataloader(
                test_map["cross1"],
                batch_size=config["batch_size"],
                sequence_length=config["sequence_length"],
                load_all_data=False,
                shuffle=shuffle,
            )
            cross2_dataloader = (
                get_dataloader(  # Shuffle and use first 5% for validation
                    test_map["cross2"],
                    batch_size=config["batch_size"],
                    sequence_length=config["sequence_length"],
                    load_all_data=False,
                    shuffle=shuffle,
                )
            )
            cross3_dataloader = (
                get_dataloader(  # Shuffle and use first 5% for validation
                    test_map["cross3"],
                    batch_size=config["batch_size"],
                    sequence_length=config["sequence_length"],
                    load_all_data=False,
                    shuffle=shuffle,
                )
            )

            dataset1 = cross1_dataloader.dataset
            dataset2 = cross2_dataloader.dataset
            dataset3 = cross3_dataloader.dataset

            combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])

            total_size = len(combined_dataset)
            indices = list(range(total_size))
            random.shuffle(indices)

            subset_size = int(0.05 * total_size)
            subset_indices = indices[:subset_size]

            subset = Subset(combined_dataset, subset_indices)

            validation_dataloader = DataLoader(
                subset,
                batch_size=cross1_dataloader.batch_size,
                collate_fn=cross1_dataloader.collate_fn,
            )

            model = TransformerTimeSeriesClassifier(
                input_features=FEATURE_SIZE,
                num_classes=CLASS_SIZE,
                seq_len=config["sequence_length"],
                d_model=config["d_model"],
                n_head=config["n_head"],
                num_encoder_layers=config["num_encoder_layers"],
                dim_feedforward=config["dim_feedforward"],
                dropout=config["dropout"],
            )

            model.to(DEVICE)

            optimizer = optimizer = torch.optim.Adam(
                model.parameters(), lr=config["lr"]
            )

            train_model(
                model,
                train_dataloader,
                eval_dataloader,
                criterion,
                optimizer,
                num_epochs=config["training_epochs"],
                device=DEVICE,
            )

            acc = test_model(validation_dataloader, model, criterion, DEVICE)

            # Send the current training result back to Tune
            train.report({"accuracy": acc})

        config = {
            "batch_size": tune.choice([2, 6, 10]),
            "sequence_length": tune.choice([10, 100, 1000]),
            "d_model": tune.choice([256, 512]),
            "n_head": tune.choice([8]),
            "num_encoder_layers": tune.choice([3, 6]),
            "dim_feedforward": tune.choice([1024]),
            "dropout": tune.choice([0.1, 0.2]),
            "training_epochs": tune.choice([3, 5, 10]),
            "lr": tune.choice([0.000001, 0.0001]),
        }
        result = tune.run(train_mnist, config=config, num_samples=20)

        best_trial = result.get_best_trial("accuracy", "max", "last")
        print(f"Best trial config: {best_trial.config}")
        print(
            f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}"
        )

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
