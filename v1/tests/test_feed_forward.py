import torch

from model.feed_forward import FeedForward

ffn = FeedForward(
    embedding_dim=32
)

x = torch.randn(
    2,
    8,
    32
)

output = ffn(x)

print(output.shape)