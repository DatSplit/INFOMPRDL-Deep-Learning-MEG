import argparse
from time import time
import torch
import torch.nn as nn
from typing import List, Tuple
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import CLASS_SIZE, DEVICE, FEATURE_SIZE, MEGDatasetType, get_dataloader
import torch.optim as optim
import torchvision
import itertools
from torch.nn import Module
import re
import math
# Hyper parameter list
## Data
sequence_length = 1
shuffle = True

## Training
training_epochs = 10
batch_size = 500
learning_rate = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam

## Model Features
hidden_size = 128
number_of_layers = 2
dropout = 0.2




class MultiHeadAttention(nn.Module):
    def __init__(self, feature_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert feature_size % num_heads == 0

        self.feature_size = feature_size
        self.num_heads = num_heads
        self.per_head_size = feature_size // num_heads

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

        # Linear transformation for output
        self.linear = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Apply linear transformations
        keys = self.key(x).view(batch_size, seq_length, self.num_heads, self.per_head_size)
        queries = self.query(x).view(batch_size, seq_length, self.num_heads, self.per_head_size)
        values = self.value(x).view(batch_size, seq_length, self.num_heads, self.per_head_size)

        # Transpose to get dimensions batch_size * num_heads * seq_length * per_head_size
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.per_head_size)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        # Transpose to get dimensions batch_size * seq_length * num_heads * per_head_size
        output = output.transpose(1, 2).contiguous()

        # Concatenate heads and apply final linear transformation
        output = output.view(batch_size, seq_length, self.feature_size)
        output = self.linear(output)

        return output, attention_weights





    
class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    
    def forward(self, x):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights
    
# Bidirectional LSTM with self-attention
class biLSTMModelSelfAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
    ):
        super(biLSTMModelSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True
        )
        self.attention = SelfAttentionLayer(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2 , num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.attention(out)
        out = self.fc(out[:, -1, :])
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)    

# Bidirectional LSTM with multi-head self-attention
class biLSTMModelMultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
    ):
        super(biLSTMModelMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True
        )
        self.attention = MultiHeadAttention(hidden_size, 4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2 , num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.attention(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
    

class LSTMModelMultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super(LSTMModelMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=False, dropout=dropout, batch_first=True
        )
        self.attention = MultiHeadAttention(hidden_size, 4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.attention(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModelMultiHeadAttentionPositionalEncoding(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super(LSTMModelMultiHeadAttentionPositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=False, dropout=dropout, batch_first=True
        )
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.attention = MultiHeadAttention(hidden_size, 4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.pos_encoder(out)
        out, _ = self.attention(out)
        out = self.dropout(out)
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
    print(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch: [{epoch+1}/{num_epochs}], Batch: [{i}/{batch_size}], Loss: {loss.item():.4f}, Time: {time() - start_time:.2f}"
                )
            if loss.item() < 0.0001:
                print(f"{loss.item()} Loss is too low, stopping training")
                return train_losses

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


def get_best_model_from_hyperparameter_tuning_result() -> Module:
    with open('hyperparameter_tuning_results.txt', 'r') as file:
        lines = file.readlines()
        best_accuracy = 0
        best_line = ""
        for line in lines:
            accuracy_match = re.search(r'Accuracy([0-9\.]+)', line)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_line = line.strip()
        print(best_line)
        return torch.load(f"checkpoints/{best_line}.pth")

def test_best_model():
    best_model= get_best_model_from_hyperparameter_tuning_result()
    test_dataloader = get_dataloader(
            MEGDatasetType.CROSS_TEST_1,
            batch_size=32,
            sequence_length=40,
            load_all_data=True,
        )
    loss_fn = nn.CrossEntropyLoss()
    return test_model(test_dataloader, best_model, loss_fn, DEVICE)

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
    training_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    loss_function = nn.CrossEntropyLoss()
    optimizer_function = torch.optim.Adam

    ## Model Features
    hidden_size = 128
    number_of_layers = 6
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

    model = LSTMModelMultiHeadAttention(
        input_size=FEATURE_SIZE,
        hidden_size=hidden_size,
        num_layers=number_of_layers,
        num_classes=CLASS_SIZE
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
        validation_split=0.15
        hyperparameters = {
            'learning_rates': [0.01],
            'batch_size': [32,64],
            'sequence_length': [10, 20, 40, 80, 160, 500],
            'dropout': [0.1,0.2,0.3,0.4,0.5]
        }

        hyperparameter_combinations = list(itertools.product(*[hyperparameters[key] for key in hyperparameters]))
        print(len(hyperparameter_combinations), hyperparameter_combinations)

        for lr, batch_size, seq_length, drop in hyperparameter_combinations:
            model = LSTMModelMultiHeadAttention(
                input_size=FEATURE_SIZE,
                hidden_size=hidden_size,
                num_layers=2,
                num_classes=CLASS_SIZE,
                dropout=drop
            )


            optimizer = optimizer_function(model.parameters(), lr=lr)

            train_dataloader = get_dataloader(
                train_map[args.hyperparameter_tuning[0]],
                batch_size=batch_size,
                sequence_length=seq_length,
                shuffle=shuffle,
                load_all_data=True 
            )

            test_dataloader, eval_dataloader = get_dataloader(
                test_map[args.hyperparameter_tuning[1]],
                batch_size=batch_size,
                sequence_length=seq_length,
                shuffle=shuffle,
                load_all_data=True,
                validation_split=validation_split
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
                file.write(f"Learning_rate{lr}Batch_size{batch_size}Sequence_length{seq_length}Dropout{drop}Test_loss{loss}Accuracy{accuracy}\n")

            torch.save(model, f"checkpoints/Learning_rate{lr}Batch_size{batch_size}Sequence_length{seq_length}Dropout{drop}Test_loss{loss}Accuracy{accuracy}.pth")

        print("Hyperparameter tuning complete")
        best_config = get_best_model_from_hyperparameter_tuning_result()
        print(f'Best hyperparameters: {best_config}')
        print(f'Validation loss, accuracy: {test_model(eval_dataloader, best_config, loss_function, DEVICE)}')

    if not args.test_set and not args.train_set and not args.hyperparameter_tuning:
        print("No Options Selected!!")


if __name__ == "__main__":
    main()
