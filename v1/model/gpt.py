import torch
import torch.nn as nn

from configs.model import ModelConfig
from model.token_embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from model.layer_norm import LayerNormalization
from model.transformer_block import TransformerBlock


class GPT(nn.Module):

    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.config = config

        self.token_embedding = TokenEmbedding(
            config.vocab_size,
            config.embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            config.max_sequence_length,
            config.embedding_dim
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    expansion_factor=config.expansion_factor,
                    dropout=config.dropout
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = LayerNormalization(
            config.embedding_dim
        )

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

    def forward(
        self,
        token_ids: torch.Tensor
    ):

        token_vectors = self.token_embedding(
            token_ids
        )

        position_vectors = self.position_embedding(
            token_ids
        )

        x = token_vectors + position_vectors

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits