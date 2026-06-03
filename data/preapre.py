import os
import pickle
import urllib.request

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(here, "input.txt")

# 1. Download the raw text (only the first time)
if not os.path.exists(input_file_path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("downloading tiny shakespeare...")
    urllib.request.urlretrieve(url, input_file_path)

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 2. Build the vocabulary: every unique character, sorted for determinism
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("unique characters:", "".join(chars))
print(f"vocab size: {vocab_size}")

# 3. The "tokenizer": two lookup tables, char <-> integer id
stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
itos = {i: ch for i, ch in enumerate(chars)}  # int -> string


def encode(s):
    return [stoi[c] for c in s]  # "Roy" -> [30, 53, 63]


def decode(ids):
    return "".join(itos[i] for i in ids)  # [30, 53, 63] -> "Roy"


# prove it's reversible BEFORE trusting it
assert decode(encode("Hello, RoyLM-0!")) == "Hello, RoyLM-0!"
print("encode/decode round-trip OK")

# 4. Split 90% train / 10% validation
n = len(data)
train_text = data[: int(n * 0.9)]
val_text = data[int(n * 0.9) :]

# 5. Encode each split into a flat stream of integer ids
train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids = np.array(encode(val_text), dtype=np.uint16)
print(f"train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")

# 6. Save the streams as raw binary (fast to memory-map during training)
train_ids.tofile(os.path.join(here, "train.bin"))
val_ids.tofile(os.path.join(here, "val.bin"))

# 7. Save the tokenizer so train.py / sample.py can decode later
with open(os.path.join(here, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)
print("wrote train.bin, val.bin, meta.pkl")
