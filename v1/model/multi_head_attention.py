import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int
    ):
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

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

        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
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

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.embedding_dim
        )

        output = self.output(output)

        return output