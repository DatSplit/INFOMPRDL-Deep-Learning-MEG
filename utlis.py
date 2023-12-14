import h5py
import numpy as np
import os
import numpy.typing as npt
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch

INTRA_TRAIN_PATH = "data/Intra/train"
INTRA_TEST_PATH = "data/Intra/test"
CROSS_TRAIN_PATH = "data/Cross/train"
CROSS_TEST_1_PATH = "data/Cross/test1"
CROSS_TEST_2_PATH = "data/Cross/test2"
CROSS_TEST_3_PATH = "data/Cross/test3"

CLASS_LABELS = ["rest", "task_motor", "task_story_math", "task_working_memory"]

FEATURE_SIZE = 248
CLASS_SIZE = 4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

label_encoder = LabelEncoder()
label_encoder.fit(CLASS_LABELS)


class MEGDataset(Dataset):
    def __init__(self, sequence_length, downsample_size=1, data_path=INTRA_TRAIN_PATH):
        self.sequence_length = sequence_length
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.downsample_size = downsample_size

        self.X, self.labels = self._collect_and_preprocess_dataset()
        self.sequences = self._create_sequences()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

    def _create_sequences(self):
        sequences = []

        for i in range(len(self.X) - self.sequence_length + 1):
            # Ensure the sequence has a consistent label
            if self.labels[i] == self.labels[i + self.sequence_length - 1]:
                sequences.append((self.X[i : i + self.sequence_length], self.labels[i]))

        return sequences

    def _collect_and_preprocess_dataset(self) -> tuple[npt.NDArray, npt.NDArray]:
        X, labels = self._collect_dataset()
        return self._preprocess_data(X, labels)

    def _preprocess_data(
        self, X: npt.NDArray, labels: list[str]
    ) -> tuple[npt.NDArray, npt.NDArray]:
        # encode labels
        encoded_labels = self._encode_labels(labels)

        # Z-Score Normalization
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)

        # downsample and return
        return (
            X_standardized[:: self.downsample_size],
            encoded_labels[:: self.downsample_size],
        )

    def _collect_dataset(self) -> tuple[npt.NDArray, list[str]]:
        X = []
        y = []
        for file_name in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file_name)

            dataset, label = self._get_dataset_and_task(file_path, file_name)

            X.append(dataset)
            y.extend([label] * len(dataset))

        return np.concatenate(X), y

    def _get_dataset_and_task(
        self, file_path: str, file_name: str
    ) -> tuple[npt.NDArray, str]:
        with h5py.File(file_path, "r") as f:
            file_name_chunks = file_name.split("_")
            dataset_name = "_".join(file_name_chunks[:-1])
            task = "_".join(file_name_chunks[:-2])
            data_matrix: npt.NDArray = f.get(dataset_name)[()]

            return data_matrix.T, task

    def _encode_labels(self, labels: list[str]) -> npt.NDArray:
        encoded_labels = label_encoder.transform(labels)
        return np.array(encoded_labels)
