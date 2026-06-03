# RoyLM-0

RoyLM-0 is a small character-level GPT trained on Tiny Shakespeare. It is meant
as a readable transformer-from-scratch project: the model learns to predict the
next character, then generates text by repeating that prediction step.

It is not a chatbot or instruction-following model. It continues text in the
style of the training data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Prepare the dataset:

```bash
python data/prepare.py
```

Train the model:

```bash
python train.py
```

Generate a sample:

```bash
python sample.py
```

Try an interactive prompt:

```bash
python prompt.py
```

Create diagnostic plots:

```bash
python visualize.py
```

## Project Layout

- `data/prepare.py` downloads Tiny Shakespeare and writes encoded train/val data.
- `model.py` defines the decoder-only transformer.
- `train.py` trains the model and saves `roylm-0.pt`.
- `sample.py` generates text from a checkpoint.
- `prompt.py` runs a simple text-continuation loop.
- `visualize.py` writes attention, embedding, and next-character plots.

Generated data, checkpoints, virtual environments, caches, and plots are ignored
by Git. Recreate them locally with the commands above.
