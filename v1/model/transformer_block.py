import torch
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNormalization


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        expansion_factor: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = LayerNormalization(
            embedding_dim
        )

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.norm2 = LayerNormalization(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            expansion_factor,
            dropout
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # Attention + Residual
        x = x + self.attention(
            self.norm1(x)
        )

        # Feed Forward + Residual
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x