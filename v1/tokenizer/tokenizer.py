import json


class CharacterTokenizer:
    def __init__(self):
        # Character -> ID
        self.char_to_id = {}

        # ID -> Character
        self.id_to_char = {}

        # Special tokens
        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

    def train(self, text):
        """
        Build the vocabulary from the given text.
        """

        # Get all unique characters
        characters = sorted(set(text))

        # Add special tokens first
        index = 0
        for token in self.special_tokens:
            self.char_to_id[token] = index
            index += 1

        # Add characters
        for char in characters:
            self.char_to_id[char] = index
            index += 1

        # Build reverse lookup dictionary
        self.id_to_char = {
            idx: char
            for char, idx in self.char_to_id.items()
        }

    def encode(self, text):
        """
        Convert text into token IDs.
        """

        tokens = []

        # Beginning of sequence
        tokens.append(self.char_to_id["<BOS>"])

        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.char_to_id["<UNK>"])

        # End of sequence
        tokens.append(self.char_to_id["<EOS>"])

        return tokens

    def decode(self, tokens):
        """
        Convert token IDs back into text.
        """

        characters = []

        for token in tokens:
            char = self.id_to_char.get(token, "<UNK>")

            # Ignore special tokens
            if char in self.special_tokens:
                continue

            characters.append(char)

        return "".join(characters)

    def save(self, filename):
        """
        Save vocabulary to a JSON file.
        """

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, indent=4, ensure_ascii=False)

    def load(self, filename):
        """
        Load vocabulary from a JSON file.
        """

        with open(filename, "r", encoding="utf-8") as f:
            self.char_to_id = json.load(f)

        self.id_to_char = {
            int(idx) if isinstance(idx, str) and idx.isdigit() else idx: char
            for idx, char in {
                v: k for k, v in self.char_to_id.items()
            }.items()
        }

    @property
    def vocab_size(self):
        """
        Return the vocabulary size.
        """
        return len(self.char_to_id)