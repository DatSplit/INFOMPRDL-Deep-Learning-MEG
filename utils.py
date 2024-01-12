from enum import Enum
import numpy as np
import os
import numpy.typing as npt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch

DATA_ROOT = "/Preprocessed"

INTRA_TRAIN_PATH = f"{DATA_ROOT}/Intra/train"
INTRA_TEST_PATH = f"{DATA_ROOT}/Intra/test"
CROSS_TRAIN_PATH = f"{DATA_ROOT}/Cross/train"
CROSS_TEST_1_PATH = f"{DATA_ROOT}/Cross/test1"
CROSS_TEST_2_PATH = f"{DATA_ROOT}/Cross/test2"
CROSS_TEST_3_PATH = f"{DATA_ROOT}/Cross/test3"

FEATURE_SIZE = 248
CLASS_SIZE = 4

TASK_TO_LABELS = {
    "rest": 0,
    "task_working_memory": 1,
    "task_motor": 2,
    "task_story_math": 3,
}

LABELS_TO_TASK = {value: key for key, value in TASK_TO_LABELS.items()}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class MEGDatasetType(Enum):
    INTRA_TRAIN = 1
    INTRA_TEST = 2
    CROSS_TRAIN = 3
    CROSS_TEST_1 = 4
    CROSS_TEST_2 = 5
    CROSS_TEST_3 = 6

torch.manual_seed(42)

def get_dataloader(
    dataset_type: MEGDatasetType,
    batch_size: int,
    shuffle: bool = False,
    sequence_length=1,
    load_all_data=False,
    validation_split=0.0,
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """
    Creates and returns a DataLoader for a specific type of MEG dataset.

    This function selects the appropriate dataset based on the specified `dataset_type`, and then creates a DataLoader using the MEGDataset class. The DataLoader handles batching, shuffling, and sequence creation from the dataset.

    Parameters:
    -----------
    dataset_type : MEGDatasetType
        An enum value specifying the type of dataset to be loaded. Different types correspond to different dataset paths.

    batch_size : int
        The size of each batch that the DataLoader should return.

    shuffle : bool, optional (default=False)
        Whether or not to shuffle the data before batching. Shuffling is typically used during training to prevent overfitting.

    sequence_length : int, optional (default=1)
        The length of the sequence to be created from the dataset. This is relevant for time-series data where sequence modeling is involved.

    load_all_data : bool, optional (default=False)
        A flag indicating whether all the data should be loaded into memory at once. Setting this to True may increase performance but requires more memory.

    validation_split: float, optional (default=0)
        If set to a decimal value will return (train_loader, validation_loader) split by the given fraction.

    Returns:
    --------
    DataLoader
        A PyTorch DataLoader instance that provides batches of data from the specified MEG dataset.

    Example Usage:
    --------------
    >>> dataloader = get_dataloader(MEGDatasetType.INTRA_TRAIN, batch_size=32, shuffle=True, sequence_length=10)
    >>> for batch in dataloader:
    >>>     process_batch(batch)
    """

    def _pad_and_collate(batch):
        (X, y) = zip(*batch)
        X = [torch.as_tensor(x) for x in X]
        y = torch.as_tensor(y)

        X_pad = pad_sequence(X, batch_first=True, padding_value=0)
        return X_pad, y

    match dataset_type:
        case MEGDatasetType.INTRA_TEST:
            training_folder_path = INTRA_TEST_PATH
        case MEGDatasetType.INTRA_TRAIN:
            training_folder_path = INTRA_TRAIN_PATH
        case MEGDatasetType.CROSS_TRAIN:
            training_folder_path = CROSS_TRAIN_PATH
        case MEGDatasetType.CROSS_TEST_1:
            training_folder_path = CROSS_TEST_1_PATH
        case MEGDatasetType.CROSS_TEST_2:
            training_folder_path = CROSS_TEST_2_PATH
        case MEGDatasetType.CROSS_TEST_3:
            training_folder_path = CROSS_TEST_3_PATH

    train_dataset = MEGDataset(
        data_path=training_folder_path,
        sequence_length=sequence_length,
        load_all_data=load_all_data,
    )

    if validation_split > 0 and validation_split < 1:
        train_dataset, validation_dataset = random_split(
            train_dataset, [1 - validation_split, validation_split]
        )
        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=_pad_and_collate,
            ),
            DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=_pad_and_collate,
            ),
        )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_pad_and_collate,
    )


class MEGDataset(Dataset):
    def __init__(self, data_path, sequence_length, load_all_data):
        """
        Sequencing through the data and creating an index of where to load it from so we are loading on the needed file at a time.
        Additionally that data is broken up into sequences, the intuition behind this is that for some model like rnn we may what
        to train and test the model on a sequence of n length data. Breaking up those sequences here still allows us to shuffle
        the data so it is not trained on the same task many time in a row.

        NOTE: Short sequences and baches will significantly increase the training time as it will be doing many many reads from your storage into ram.
            To speed this up train on longer sequences or larger bathers.

        `load_all_data`: Use this to load all the data into RAM if you can spare it. Will make training much faster.
        """
        
        self._data_path = data_path
        self._files = [file_name for file_name in os.listdir(data_path)]

        # Below are used it you load all data into RAM
        self._data = []
        self._labels = []
        self._load_all_data = load_all_data

        # Sequences will be a list of tuple(tuple(start, end), file_index)
        # I think this approach is a little convoluted and I will try to clean it up but for now it works :)
        self._sequences: tuple[tuple[int, int], int] = []
        for i, file_name in enumerate(self._files):
            file_path = os.path.join(self._data_path, file_name)
            dataset = np.load(file_path)

            data_size = dataset["data"].shape[0]
            for start in range(0, data_size, sequence_length):
                end = min(start + sequence_length, data_size)

                self._sequences.append(((start, end), i))

            if load_all_data:
                self._data.append(dataset["data"].reshape(-1, 248))
                self._labels.append(dataset["label"].item())

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, index):
        sequence = self._sequences[index]
        file_name = self._files[sequence[1]]
        start = sequence[0][0]
        end = sequence[0][1]
        if self._load_all_data:
            return self._load_sequence_data_from_RAM(start, end, sequence[1])
        return self._load_sequence_data_from_file(start, end, file_name)

    def _load_sequence_data_from_file(
        self, start: int, end: int, file_name: str
    ) -> tuple[npt.NDArray, npt.NDArray]:
        file_path = os.path.join(self._data_path, file_name)
        dataset = np.load(file_path)
        X = dataset["data"].reshape(-1, 248)
        y = dataset["label"].item()
        return X[start:end], y

    def _load_sequence_data_from_RAM(
        self, start: int, end: int, data_index: int
    ) -> tuple[npt.NDArray, npt.NDArray]:
        X = self._data[data_index]
        y = self._labels[data_index]
        return X[start:end], y
