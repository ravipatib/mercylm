"""
Data preparation for Mercy: The Only Human Left with Pigeon Gerald.

Trains a BPE tokenizer on the generated conversations, then converts
train.json and val.json into tokenized tensor files (train.pt, val.pt).

After preparation, writes the actual vocab size back into a
vocab_size.txt file so training always uses the correct value
without manual config editing.

Special tokens:
  [USER]  — marks the human turn
  [MERCY] — marks Mercy's response
  [END]   — marks end of response
"""

import json
import os
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from .config import TLHAConfig

USER_TOKEN  = "[USER]"
CHAR_TOKEN  = "[MERCY]"
END_TOKEN   = "[END]"
SPECIAL_TOKENS = [USER_TOKEN, CHAR_TOKEN, END_TOKEN]


def build_corpus(records: list[dict]) -> list[str]:
    corpus = []
    for r in records:
        corpus.append(r["input"])
        corpus.append(r["response"])
    return corpus


def train_tokenizer(corpus: list[str], vocab_size: int, save_path: str) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]"] + SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.save(save_path)
    actual = tokenizer.get_vocab_size()
    print(f"Tokenizer saved → {save_path}  (vocab size: {actual})")
    return tokenizer


def encode_records(
    records: list[dict],
    tokenizer: Tokenizer,
    context_len: int,
) -> torch.Tensor:
    """
    Format: [USER] input [MERCY] response [END]
    Padded to context_len with -1 (ignored in loss).
    -1 in x positions is replaced with 0 in dataset.py before embedding.
    """
    user_id = tokenizer.token_to_id(USER_TOKEN)
    char_id = tokenizer.token_to_id(CHAR_TOKEN)
    end_id  = tokenizer.token_to_id(END_TOKEN)

    all_samples = []
    for r in records:
        inp  = tokenizer.encode(r["input"]).ids
        resp = tokenizer.encode(r["response"]).ids
        seq  = [user_id] + inp + [char_id] + resp + [end_id]
        seq  = seq[:context_len]
        seq  = seq + [-1] * (context_len - len(seq))
        all_samples.append(seq)

    return torch.tensor(all_samples, dtype=torch.long)


def prepare(cfg: TLHAConfig):
    data_dir   = cfg.train.data_dir
    tok_path   = os.path.join(data_dir, "tokenizer.json")
    train_pt   = os.path.join(data_dir, "train.pt")
    val_pt     = os.path.join(data_dir, "val.pt")
    vocab_file = os.path.join(data_dir, "vocab_size.txt")

    with open(os.path.join(data_dir, "train.json")) as f:
        train_records = json.load(f)
    with open(os.path.join(data_dir, "val.json")) as f:
        val_records = json.load(f)

    print(f"Loaded {len(train_records):,} train + {len(val_records):,} val records")

    all_records = train_records + val_records
    corpus = build_corpus(all_records)
    tokenizer = train_tokenizer(corpus, cfg.model.vocab_size, tok_path)

    # Save actual vocab size so train.py picks it up automatically
    actual_vocab = tokenizer.get_vocab_size()
    with open(vocab_file, "w") as f:
        f.write(str(actual_vocab))
    print(f"Vocab size saved → {vocab_file}  ({actual_vocab} tokens)")

    print("Encoding training data...")
    train_tensor = encode_records(train_records, tokenizer, cfg.model.context_len)
    print("Encoding validation data...")
    val_tensor   = encode_records(val_records,   tokenizer, cfg.model.context_len)

    torch.save(train_tensor, train_pt)
    torch.save(val_tensor,   val_pt)
    print(f"train.pt: {train_tensor.shape}")
    print(f"val.pt:   {val_tensor.shape}")
    print("\nData preparation complete. Ready to train.")


if __name__ == "__main__":
    cfg = TLHAConfig()
    prepare(cfg)
