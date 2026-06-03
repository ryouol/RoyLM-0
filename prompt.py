# Interactive text continuation with RoyLM-0.
import os

import torch

from model import GPT

here = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(here, "roylm-0.pt")

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

model = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()
model.to(device)

temperature = 0.8
max_new_tokens = 300

print("RoyLM-0 - type the start of some text and it will continue it.")
print("It writes Shakespeare-style text; it does NOT answer questions.")
print("Try: ROMEO:    or just press Enter. Ctrl-C to quit.\n")

while True:
    try:
        prompt = input("you > ")
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        break

    unknown = sorted({c for c in prompt if c not in stoi})
    if unknown:
        print(f"  (ignoring characters not in the vocab: {unknown})")
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        ids = [stoi["\n"]]

    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    y = model.generate(x, max_new_tokens, temperature=temperature)
    print("roylm-0 >", "".join(itos[i] for i in y[0].tolist()), "\n")
