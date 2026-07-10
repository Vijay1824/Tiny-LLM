import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    Input:
        (batch_size, sequence_length)

    Output:
        (batch_size, sequence_length, embedding_dim)
    """

    def __init__(
        self,
        max_sequence_length: int,
        embedding_dim: int
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim
        )

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:

        batch_size, sequence_length = token_ids.shape

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            sequence_length
        )

        return self.position_embedding(positions)