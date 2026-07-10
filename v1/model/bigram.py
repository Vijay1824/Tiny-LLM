import torch
import torch.nn as nn


class BigramLanguageModel(nn.Module):
    """
    A simple Bigram Language Model.

    Given the current token, it predicts the next token.

    Architecture:

    Token
      │
      ▼
    Embedding Table
      │
      ▼
    Logits
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size

        # Each token directly learns scores for every possible next token.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size
        )

    def forward(self, input_tokens):
        """
        Parameters
        ----------
        input_tokens : Tensor
            Shape: (batch_size)

        Returns
        -------
        logits : Tensor
            Shape: (batch_size, vocab_size)
        """

        logits = self.embedding(input_tokens)

        return logits

    def generate(self, start_token, max_new_tokens=20):
        """
        Generate tokens one by one.
        """

        generated = [start_token]

        current = torch.tensor([start_token])

        for _ in range(max_new_tokens):

            logits = self.forward(current)

            next_token = torch.argmax(
                logits,
                dim=-1
            )

            token = next_token.item()

            generated.append(token)

            current = next_token

        return generated