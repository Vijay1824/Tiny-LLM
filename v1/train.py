from tokenizer.tokenizer import CharacterTokenizer
from data.dataset import TextDataset
from model.bigram import BigramLanguageModel
from training.trainer import Trainer
from training.checkpoint import save_checkpoint
import torch

torch.backends.cudnn.benchmark = True

text = open(
    "data/sample.txt",
    encoding="utf-8"
).read()

tokenizer = CharacterTokenizer()

tokenizer.train(text)

dataset = TextDataset(
    text,
    tokenizer
)

model = BigramLanguageModel(
    tokenizer.vocab_size
)

trainer = Trainer(
    model,
    dataset,
    learning_rate=1e-2,
    batch_size=256
)

trainer.train(
    epochs=100
)

save_checkpoint(model)