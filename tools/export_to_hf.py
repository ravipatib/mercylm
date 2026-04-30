"""
Export Mercy: The Only Human Left with Pigeon Gerald to HuggingFace Hub.

Uploads:
  1. The model weights + tokenizer → your-username/mercy-15M
  2. The dataset (train.json + val.json) → your-username/mercy-80k

Usage:
    python tools/export_to_hf.py --model-repo your-username/mercy-15M \
                                  --data-repo  your-username/mercy-80k \
                                  --token      hf_xxxxx
"""

import argparse
import json
import os
import shutil
import torch

from huggingface_hub import HfApi, login, create_repo


# ── model card ────────────────────────────────────────────────────────────────

MODEL_CARD = """\
---
language: en
license: mit
tags:
  - tiny-model
  - character-llm
  - from-scratch
  - pytorch
  - language-model
---

# Mercy: The Only Human Left with Pigeon Gerald 15M

A ~15M parameter language model trained to speak as a human who woke up one saturday and everyone was gone.
Alone for 847 days. Pigeon named Gerald. Still counting sunsets.

## Usage

```python
pip install tlha-llm
python -m tlha chat
```

Or in Python:

```python
from tlha.inference import load_model_hf, chat_loop
model, tokenizer, _ = load_model_hf("your-username/mercy-15M", device="cpu")
chat_loop(model, tokenizer, device="cpu")
```

## Sample

```
You> hello
Mercy> oh. you answered. it's day 847.

You> goodnight
Mercy> day 847 done. goodnight out there. goodnight gerald.
```

## Architecture

| | |
|---|---|
| Parameters | ~15M |
| Layers | 10 |
| Hidden dim | 384 |
| Heads | 8 |
| Vocab | 4,096 BPE |
| Context | 128 tokens |

Trained from scratch on 80K synthetic conversations. Vanilla transformer.
No fine-tuning. No pretrained weights. Just next-token prediction.
"""


# ── dataset card ──────────────────────────────────────────────────────────────

DATASET_CARD = """\
---
language: en
license: mit
tags:
  - synthetic
  - character-llm
  - conversation
---

# Mercy: The Only Human Left with Pigeon Gerald — 80K Dataset

80,000 synthetic single-turn conversations for training Mercy: The Only Human Left with Pigeon Gerald.

The character: a human who woke up one saturday and everyone was gone, alone for 840-1100 days, living in
an unnamed city, talking to a pigeon named Gerald.

## Format

```json
{"input": "hello", "response": "oh. you answered. it's day 847.", "topic": "greeting"}
```

## Stats

| | |
|---|---|
| Total samples | 80,000 |
| Train split | 57,000 |
| Val split | 3,000 |
| Topics | 60 |
| Generation | Synthetic template composition |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("your-username/mercy-80k")
print(ds["train"][0])
```
"""


# ── export functions ──────────────────────────────────────────────────────────

def export_model(
    model_repo: str,
    checkpoint_path: str,
    tokenizer_path: str,
    api: HfApi,
):
    print(f"\nExporting model to {model_repo} ...")
    create_repo(model_repo, exist_ok=True, repo_type="model")

    export_dir = "hf_export/model"
    os.makedirs(export_dir, exist_ok=True)

    # Load checkpoint and save state dict only (smaller file)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    torch.save(ckpt["model_state_dict"], os.path.join(export_dir, "model.pt"))

    # Save config as JSON
    import json, dataclasses
    cfg_dict = {
        "model": dataclasses.asdict(cfg.model),
        "character_name": cfg.character_name,
        "character_tag": cfg.character_tag,
    }
    with open(os.path.join(export_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Copy tokenizer
    shutil.copy(tokenizer_path, os.path.join(export_dir, "tokenizer.json"))

    # Write model card
    with open(os.path.join(export_dir, "README.md"), "w") as f:
        f.write(MODEL_CARD.replace("your-username", model_repo.split("/")[0]))

    # Upload
    api.upload_folder(
        folder_path=export_dir,
        repo_id=model_repo,
        repo_type="model",
    )
    print(f"Model uploaded → https://huggingface.co/{model_repo}")


def export_dataset(
    data_repo: str,
    data_dir: str,
    api: HfApi,
):
    print(f"\nExporting dataset to {data_repo} ...")
    create_repo(data_repo, exist_ok=True, repo_type="dataset")

    export_dir = "hf_export/dataset"
    os.makedirs(export_dir, exist_ok=True)

    shutil.copy(os.path.join(data_dir, "train.json"), os.path.join(export_dir, "train.json"))
    shutil.copy(os.path.join(data_dir, "val.json"),   os.path.join(export_dir, "val.json"))

    with open(os.path.join(export_dir, "README.md"), "w") as f:
        f.write(DATASET_CARD.replace("your-username", data_repo.split("/")[0]))

    api.upload_folder(
        folder_path=export_dir,
        repo_id=data_repo,
        repo_type="dataset",
    )
    print(f"Dataset uploaded → https://huggingface.co/datasets/{data_repo}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export TLHA to HuggingFace Hub")
    parser.add_argument("--model-repo",  default="your-username/mercy-15M")
    parser.add_argument("--data-repo",   default="your-username/mercy-80k")
    parser.add_argument("--checkpoint",  default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer",   default="data/tokenizer.json")
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--token",       default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--model-only",  action="store_true")
    parser.add_argument("--dataset-only",action="store_true")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: provide --token or set HF_TOKEN environment variable")
        return

    login(token=token)
    api = HfApi()

    if not args.dataset_only:
        export_model(args.model_repo, args.checkpoint, args.tokenizer, api)

    if not args.model_only:
        export_dataset(args.data_repo, args.data_dir, api)

    print("\nDone. Now update the HF repo links in README.md and inference.py.")


if __name__ == "__main__":
    main()
