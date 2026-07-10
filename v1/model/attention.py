import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head causal self-attention.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.embedding_dim
        )

        sequence_length = x.size(1)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention,
            V
        )

        return output