import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Dataset for sequence-based next-token prediction.
    """

    def __init__(self, text, tokenizer, context_length=32):
        self.tokens = tokenizer.encode(text)
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.context_length]
        y = self.tokens[idx + 1:idx + self.context_length + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

'''
class TextDataset(Dataset):
    """
    A simple dataset for next-token prediction.

    Given a token sequence:
    [BOS, h, e, l, l, o, EOS]

    It produces:

    Input -> Target

    BOS -> h
    h   -> e
    e   -> l
    l   -> l
    l   -> o
    o   -> EOS
    """

    def __init__(self, text, tokenizer):
        """
        Parameters
        ----------
        text : str
            Raw training text.

        tokenizer : CharacterTokenizer
            The tokenizer used to convert text into token IDs.
        """

        self.tokenizer = tokenizer

        # Convert text into token IDs
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        """
        Number of training samples.
        """

        return len(self.tokens) - 1

    def __getitem__(self, index):
        """
        Returns one training sample.

        Example

        tokens:
        [2,7,6,8,8,9,3]

        index = 0

        returns

        input = 2
        target = 7
        """

        input_token = self.tokens[index]
        target_token = self.tokens[index + 1]

        return (
            torch.tensor(input_token, dtype=torch.long),
            torch.tensor(target_token, dtype=torch.long),
        )
'''