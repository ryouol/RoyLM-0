# Generate text from a trained checkpoint.
import os

import torch

from model import GPT

here = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(here, "roylm-0.pt")

start = "\n"
max_new_tokens = 500
temperature = 0.8
top_k = None

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
config = ckpt["config"]
meta = ckpt["meta"]
stoi, itos = meta["stoi"], meta["itos"]


def encode(text):
    return [stoi[c] for c in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


model = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()
model.to(device)

ids = encode(start)
x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
print(decode(y[0].tolist()))
