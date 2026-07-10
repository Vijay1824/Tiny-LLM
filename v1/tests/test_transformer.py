import torch

from model.transformer_block import TransformerBlock

block = TransformerBlock(
    embedding_dim=64,
    num_heads=8
)

x = torch.randn(
    2,
    16,
    64
)

output = block(x)

print(output.shape)