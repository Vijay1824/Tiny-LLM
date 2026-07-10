import torch

from tokenizer.tokenizer import CharacterTokenizer
from data.dataset import SequenceDataset
from configs.model import ModelConfig
from model.gpt import GPT
from training.trainer import GPTTrainer


def main():

    # Better CUDA performance
    torch.set_float32_matmul_precision("high")

    # Load training text
    with open(
        "data/sample.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    print(f"Loaded dataset: {len(text):,} characters")

    # Train tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)

    print(f"Vocabulary Size: {tokenizer.vocab_size}")

    # Dataset
    dataset = SequenceDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=64
    )

    # Model configuration
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=64,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        expansion_factor=4,
        dropout=0.1,
    )

    model = GPT(config)

    trainer = GPTTrainer(
        model=model,
        dataset=dataset,
        learning_rate=3e-4,
        batch_size=32,
    )

    trainer.train(epochs=2)


if __name__ == "__main__":
    main()