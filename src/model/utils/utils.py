import math

import torch
from torch.nn import (
    Module,
    Dropout,
)


class PositionalEncoding(Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class Mask:
    def __init__(self, device):
        self.device = device

    def padding(self, x, pad=0):
        mask = x == pad
        return torch.where(mask, -torch.inf, pad).to(self.device)

    def subsequent(self, y):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        size = y.size(-1)
        return torch.triu(
            torch.full((size, size), float("-inf")), diagonal=1
        ).to(self.device)
