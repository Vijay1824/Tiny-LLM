import torch

from model.positional_embedding import PositionalEmbedding

EMBED_DIM = 16

MAX_LENGTH = 32

embedding = PositionalEmbedding(
    MAX_LENGTH,
    EMBED_DIM
)

tokens = torch.tensor([
    [1,2,3,4],
    [5,6,7,8]
])

output = embedding(tokens)

print(output.shape)