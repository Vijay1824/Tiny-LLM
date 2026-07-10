import torch

from model.multi_head_attention import MultiHeadAttention

attention = MultiHeadAttention(
    embedding_dim=32,
    num_heads=4
)

x = torch.randn(
    2,
    8,
    32
)

output = attention(x)

print(output.shape)