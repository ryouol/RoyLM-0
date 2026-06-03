# Write a few simple diagnostic plots for a trained checkpoint.
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import GPT

here = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(here, "roylm-0.pt")

prompt = "ROMEO:"
layer = -1

device = "cpu"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
config, meta = ckpt["config"], ckpt["meta"]
stoi, itos = meta["stoi"], meta["itos"]

model = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()
model.to(device)

ids = [stoi[c] for c in prompt if c in stoi]
chars = [itos[i] for i in ids]
x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    logits, _ = model(x)

probs = F.softmax(logits[0, -1], dim=-1)
top = torch.topk(probs, 15)
labels = [repr(itos[i.item()]) for i in top.indices]
plt.figure(figsize=(8, 4))
plt.bar(range(len(labels)), top.values.numpy(), color="#4C72B0")
plt.xticks(range(len(labels)), labels, rotation=45)
plt.ylabel("probability")
plt.title(f"P(next character)  after  {prompt!r}")
plt.tight_layout()
plt.savefig(os.path.join(here, "viz_nextchar.png"), dpi=120)
print("saved viz_nextchar.png")

att = model.transformer.h[layer].attn.last_att[0]
nh = att.shape[0]
fig, axes = plt.subplots(1, nh, figsize=(3.2 * nh, 3.4))
if nh == 1:
    axes = [axes]
for h in range(nh):
    ax = axes[h]
    ax.imshow(att[h].numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"head {h}", fontsize=10)
    ax.set_xticks(range(len(chars)), chars)
    ax.set_yticks(range(len(chars)), chars)
    if h == 0:
        ax.set_ylabel("query (from char)")
    ax.set_xlabel("key (looks at)")
fig.suptitle(f"attention  layer {layer}  |  {prompt!r}  "
             f"(causal mask)")
fig.tight_layout()
fig.savefig(os.path.join(here, "viz_attention.png"), dpi=120)
print("saved viz_attention.png")

W = model.transformer.wte.weight.detach().numpy()
Wc = W - W.mean(0, keepdims=True)
_, _, Vt = np.linalg.svd(Wc, full_matrices=False)
xy = Wc @ Vt[:2].T
plt.figure(figsize=(8, 8))
plt.scatter(xy[:, 0], xy[:, 1], s=12, alpha=0.25, color="#C44E52")
for i in range(len(itos)):
    plt.annotate(repr(itos[i]), (xy[i, 0], xy[i, 1]), fontsize=9)
plt.title("learned character embeddings (PCA to 2D)")
plt.tight_layout()
plt.savefig(os.path.join(here, "viz_embeddings.png"), dpi=120)
print("saved viz_embeddings.png")

if "agg" not in plt.get_backend().lower():
    plt.show()
