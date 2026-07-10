import torch

from model.attention import SelfAttention

attention = SelfAttention(
    embedding_dim=16
)

x = torch.randn(
    2,
    8,
    16
)

output = attention(x)

print(output.shape)