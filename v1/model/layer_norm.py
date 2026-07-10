import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Wrapper around PyTorch LayerNorm.
    """

    def __init__(
        self,
        embedding_dim: int
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            embedding_dim
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.norm(x)