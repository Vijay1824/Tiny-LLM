import torch

from configs.model import ModelConfig
from model.gpt import GPT

config = ModelConfig(
    vocab_size=100
)

model = GPT(config)

tokens = torch.randint(
    0,
    config.vocab_size,
    (2,16)
)

output = model(tokens)

print(output.shape)