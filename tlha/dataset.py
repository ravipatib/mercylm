"""
Dataset and DataLoader for Mercy: The Only Human Left with Pigeon Gerald.

Loads pre-tokenized tensors and serves (input, target) pairs
for next-token prediction training.

MPS notes:
  - num_workers=0 is required for MPS (multiprocessing + MPS = crash)
  - pin_memory=False for MPS (pin_memory is CUDA-only)
  - persistent_workers=False for same reason
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

from .config import TLHAConfig


class ConversationDataset(Dataset):
    """
    Each sample is one conversation of shape (context_len,).
    Returns (x, y) where y = x shifted left by 1 (next-token targets).

    Padding value -1 is used in y positions (ignored by cross-entropy loss).
    In x positions, -1 is replaced with 0 (a valid token id) so the
    embedding lookup never receives an out-of-range index.
    """

    def __init__(self, tensor_path: str):
        self.data = torch.load(tensor_path, weights_only=True)  # (N, T)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        seq = self.data[idx]          # (T,)
        x   = seq[:-1].clone()        # tokens 0..T-2  — model input
        y   = seq[1:].clone()         # tokens 1..T-1  — prediction targets

        # Replace padding in x with token 0 — safe for embedding lookup.
        # Corresponding y values are -1 so loss ignores these positions.
        x[x == -1] = 0

        return x, y


def get_loaders(cfg: TLHAConfig):
    """Build train and validation DataLoaders tuned for Apple Silicon."""
    train_path = os.path.join(cfg.train.data_dir, "train.pt")
    val_path   = os.path.join(cfg.train.data_dir, "val.pt")

    train_ds = ConversationDataset(train_path)
    val_ds   = ConversationDataset(val_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,        # MPS requirement — must be 0
        pin_memory=False,     # pin_memory is CUDA-only, not MPS
        persistent_workers=False,
        drop_last=True,       # avoids uneven last batch issues on MPS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    return train_loader, val_loader
