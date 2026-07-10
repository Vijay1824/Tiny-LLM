from tokenizer.tokenizer import CharacterTokenizer

text = "hello world"

tokenizer = CharacterTokenizer()

# Build vocabulary
tokenizer.train(text)

print("Vocabulary:")
print(tokenizer.char_to_id)

# Encode
tokens = tokenizer.encode(text)
print("\nEncoded:")
print(tokens)

# Decode
decoded = tokenizer.decode(tokens)
print("\nDecoded:")
print(decoded)

# Vocabulary size
print("\nVocabulary Size:")
print(tokenizer.vocab_size)

# Save vocabulary
tokenizer.save("vocab.json")

# Load vocabulary into a new tokenizer
new_tokenizer = CharacterTokenizer()
new_tokenizer.load("vocab.json")

print("\nLoaded Vocabulary:")
print(new_tokenizer.char_to_id)

print("\nEncoding after loading:")
print(new_tokenizer.encode("hello"))