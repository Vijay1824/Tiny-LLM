from dataclasses import dataclass


@dataclass
class ModelConfig:

    vocab_size: int

    max_sequence_length: int = 128

    embedding_dim: int = 128

    num_heads: int = 4

    num_layers: int = 4

    expansion_factor: int = 4

    dropout: float = 0.1