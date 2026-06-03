# Train RoyLM-0 on Tiny Shakespeare.
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig

batch_size = 32
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000
eval_interval = 250
eval_iters = 50
grad_clip = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CKPT_PATH = os.path.join(HERE, "roylm-0.pt")

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"using device: {device}")
torch.manual_seed(1337)

with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
print(f"vocab_size = {vocab_size}")

train_data = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)
val_data = np.fromfile(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Targets are the same text shifted one character to the right.
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
)
model = GPT(config).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


best_val = float("inf")
t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| {time.time() - t0:.1f}s")
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "config": config, "meta": meta}, CKPT_PATH)
            print(f"  -> saved {CKPT_PATH}")

    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

print("done.")
