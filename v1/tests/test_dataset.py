from tokenizer.tokenizer import CharacterTokenizer
from data.dataset import TextDataset

text = "hello"

tokenizer = CharacterTokenizer()
tokenizer.train(text)

dataset = TextDataset(text, tokenizer)

print("Vocabulary")
print(tokenizer.char_to_id)

print()

print("Encoded Tokens")
print(tokenizer.encode(text))

print()

print("Dataset Length")
print(len(dataset))

print()

print("Samples")

for i in range(len(dataset)):
    input_token, target_token = dataset[i]

    print(
        f"Input: {input_token.item()} "
        f"Target: {target_token.item()}"
    )