import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed Forward Network (MLP).

    Architecture:

    Input
      │
      ▼
    Linear
      │
      ▼
    GELU
      │
      ▼
    Linear
      │
      ▼
    Output
    """

    def __init__(
        self,
        embedding_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        hidden_dim = embedding_dim * expansion_factor

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            ),

            nn.Dropout(dropout)

        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.network(x)