import os
import pickle
import urllib.request

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(HERE, "input.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


if not os.path.exists(INPUT_PATH):
    print("downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(DATA_URL, INPUT_PATH)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = f.read()

print(f"length of dataset in characters: {len(data):,}")

chars = sorted(set(data))
vocab_size = len(chars)
print("unique characters:", "".join(chars))
print(f"vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[c] for c in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


sample = data[:1000]
assert decode(encode(sample)) == sample
print("encode/decode round trip OK")

n = len(data)
train_text = data[: int(n * 0.9)]
val_text = data[int(n * 0.9) :]

train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids = np.array(encode(val_text), dtype=np.uint16)
print(f"train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")

train_ids.tofile(os.path.join(HERE, "train.bin"))
val_ids.tofile(os.path.join(HERE, "val.bin"))

with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

print("wrote train.bin, val.bin, meta.pkl")
