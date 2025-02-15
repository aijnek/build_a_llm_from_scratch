import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) 

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# read the text file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# split the text into words
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# extract unique words
all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])

vocab_size = len(all_words)
print(f"vocab size: {vocab_size}")

# create a dictionary(vocabulary) that maps words to integers
vocab = {token:integer for integer,token in enumerate(all_words)}

# tokenizer instance based on the vocab created above
tokenizer = SimpleTokenizerV2(vocab)

text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

decoded_text = tokenizer.decode(ids)
print(decoded_text)
