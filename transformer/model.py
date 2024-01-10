import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerTimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        input_features,
        num_classes,
        seq_len,
        d_model=512,
        n_head=8,
        num_encoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(TransformerTimeSeriesClassifier, self).__init__()
        self.d_model = d_model

        self.linear = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_encoder_layers,
        )
        # self.out = nn.Linear(d_model, num_classes)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, num_classes),
            # nn.ReLU(),
            # nn.Linear(512, 128),
            # nn.ReLU(),
            # nn.Linear(128, num_classes),
        )

    def forward(self, src):
        src = self.linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.out(output)
        return F.log_softmax(output, dim=1)
