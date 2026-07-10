import torch

from model.token_embedding import TokenEmbedding

VOCAB_SIZE = 100

EMBED_DIM = 16

embedding = TokenEmbedding(
    VOCAB_SIZE,
    EMBED_DIM
)

tokens = torch.tensor([
    [1,2,3,4],
    [5,6,7,8]
])

output = embedding(tokens)

print(output.shape)

print(output)