"""
Configuration for Mercy: The Only Human Left with Pigeon Gerald.

A ~15M parameter language model trained to speak as a human who woke up one saturday and everyone was gone —
alone for 847 days, with a pigeon named Gerald and not much else.

Architecture is intentionally vanilla: this project is about the character,
not the engineering. Standard transformer keeps the code readable and
training reproducible on any consumer GPU in under 20 minutes.

Optimised for Apple Silicon M-series via MPS.
"""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class ModelConfig:
    # Vocabulary and sequence
    vocab_size: int = 4096        # overwritten automatically by prepare from vocab_size.txt
    context_len: int = 128        # tokens per sample

    # Transformer dimensions — tuned for ~15M params
    embed_dim: int = 384
    num_heads: int = 8            # embed_dim must be divisible by num_heads
    num_layers: int = 10
    ffn_dim: int = 1024
    dropout: float = 0.1

    # Derived — recomputed by __post_init__
    head_dim: int = field(init=False)

    def __post_init__(self):
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        self.head_dim = self.embed_dim // self.num_heads


@dataclass
class TrainConfig:
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    # Batch — 128 is comfortable on Apple Silicon, drop to 64 if OOM
    batch_size: int = 128
    num_epochs: int = 20

    # Optimiser
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # LR schedule — cosine decay with warmup
    warmup_steps: int = 500

    # Logging
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 500

    # Device — "auto" detects MPS on Apple Silicon automatically
    device: str = "auto"


@dataclass
class TLHAConfig:
    """Top-level config bundling model + training settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    character_name: str = "Mercy"
    character_tag: str = "mercy"

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "model": asdict(self.model),
                "train": asdict(self.train),
                "character_name": self.character_name,
                "character_tag": self.character_tag,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TLHAConfig":
        with open(path) as f:
            raw = json.load(f)
        cfg = cls()
        for k, v in raw["model"].items():
            if k != "head_dim" and hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
        cfg.model.__post_init__()
        for k, v in raw["train"].items():
            if hasattr(cfg.train, k):
                setattr(cfg.train, k, v)
        cfg.character_name = raw.get("character_name", cfg.character_name)
        cfg.character_tag  = raw.get("character_tag",  cfg.character_tag)
        return cfg


def resolve_device(preference: str = "auto") -> str:
    """
    Pick the best available compute device.
    Priority: cuda > mps (Apple Silicon) > cpu
    """
    import torch
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_vocab_size(cfg: TLHAConfig) -> int:
    """
    Read the actual vocab size from the trained tokenizer and
    update cfg.model.vocab_size in place. Call this before
    building the model so config always matches the tokenizer.
    Returns the actual vocab size.
    """
    import os
    from tokenizers import Tokenizer
    tok_path = os.path.join(cfg.train.data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. "
            "Run `python -m tlha prepare` first."
        )
    tok = Tokenizer.from_file(tok_path)
    actual = tok.get_vocab_size()
    cfg.model.vocab_size = actual
    cfg.model.__post_init__()
    return actual
