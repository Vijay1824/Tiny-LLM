import torch

from model.bigram import BigramLanguageModel

VOCAB_SIZE = 12

model = BigramLanguageModel(VOCAB_SIZE)

sample = torch.tensor([2])

output = model(sample)

print(output.shape)

generated = model.generate(2, 10)

print(generated)