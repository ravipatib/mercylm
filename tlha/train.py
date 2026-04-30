"""
Training loop for Mercy: The Only Human Left with Pigeon Gerald.

Optimised for Apple Silicon via MPS. Tested on Apple Silicon.
Also works on CUDA and CPU.

MPS-specific decisions based on current PyTorch MPS status (2025):
  - fused=False for AdamW — fused Adam has known silent bugs on MPS
    (addcmul_/addcdiv_ on non-contiguous tensors, fixed in macOS 15+
    but safer to leave off universally)
  - PYTORCH_ENABLE_MPS_FALLBACK=1 set at startup — handles any ops
    not yet implemented in Metal kernels
  - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 set at startup — disables
    MPS memory cap that can cause OOM on large batches
  - float32 throughout — MPS AMP is still inconsistent in 2025
  - periodic torch.mps.empty_cache() — prevents memory fragmentation
  - num_workers=0 in DataLoader — MPS + multiprocessing = crash
"""

import os
import math
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from .config import TLHAConfig, resolve_device
from .model import MercyLLM
from .dataset import get_loaders


def _set_mps_env():
    """
    Set environment variables required for stable MPS training.
    Must be called before any MPS tensors are created.
    """
    # Allow ops not yet in Metal to fall back to CPU silently
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Disable MPS memory cap — safe on Apple Silicon unified memory
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def load_actual_vocab_size(data_dir: str) -> int:
    """
    Read vocab size written by prepare_data.py.
    Raises a clear error if prepare hasn't been run yet.
    """
    vocab_file = os.path.join(data_dir, "vocab_size.txt")
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(
            f"\nvocab_size.txt not found in {data_dir}/\n"
            "Run this first:  python -m tlha prepare\n"
        )
    with open(vocab_file) as f:
        return int(f.read().strip())


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 10% of peak LR."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def mps_cache_flush(device: str, step: int, every: int = 200):
    """Periodically free MPS cache to prevent memory fragmentation."""
    if device == "mps" and step % every == 0:
        torch.mps.empty_cache()


@torch.no_grad()
def evaluate(model: MercyLLM, loader, device: str) -> float:
    model.eval()
    total_loss, steps = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        steps += 1
    model.train()
    return total_loss / max(steps, 1)


def train(cfg: TLHAConfig):
    # ── MPS environment setup — must happen before device init ────────────────
    _set_mps_env()

    # ── device ────────────────────────────────────────────────────────────────
    device = resolve_device(cfg.train.device)
    cfg.train.device = device
    print(f"Device:     {device}")
    if device == "mps":
        print(f"            Apple Silicon MPS")
        print(f"            PYTORCH_ENABLE_MPS_FALLBACK = "
              f"{os.environ['PYTORCH_ENABLE_MPS_FALLBACK']}")
        print(f"            PYTORCH_MPS_HIGH_WATERMARK_RATIO = "
              f"{os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']}")

    # ── vocab size — always read from tokenizer, never trust default ──────────
    actual_vocab = load_actual_vocab_size(cfg.train.data_dir)
    cfg.model.vocab_size = actual_vocab
    cfg.model.__post_init__()
    print(f"Vocab size: {actual_vocab}")

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_loaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * cfg.train.num_epochs
    print(f"Train steps/epoch: {steps_per_epoch}")
    print(f"Total steps:       {total_steps:,}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = MercyLLM(cfg.model).to(device)
    print(f"Parameters:        {model.num_parameters():,}")

    # ── optimiser ─────────────────────────────────────────────────────────────
    # fused=False — known MPS bug with addcmul_/addcdiv_ on non-contiguous
    # tensors (silent weight freezing). Safe on all devices. No perf cost
    # at 15M params where the optimizer step is not a bottleneck.
    optimiser = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        fused=False,
    )

    scheduler = LambdaLR(
        optimiser,
        lr_lambda=lambda s: cosine_with_warmup(
            s, cfg.train.warmup_steps, total_steps
        ),
    )

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    log_dir = os.path.join("logs", f"mercy_{time.strftime('%Y%m%d_%H%M%S')}")
    writer  = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: tensorboard --logdir logs")
    print(f"             then open http://localhost:6006")

    best_val_loss = float("inf")
    no_improve    = 0
    patience      = 5          # stop if val loss doesn't improve for 5 evals
    global_step   = 0
    t0            = time.time()

    print(f"\nTraining Mercy for {cfg.train.num_epochs} epochs...")
    print("─" * 60)

    for epoch in range(1, cfg.train.num_epochs + 1):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimiser.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip
            )
            optimiser.step()
            scheduler.step()
            global_step += 1

            # Flush MPS cache periodically
            mps_cache_flush(device, global_step)

            # Log
            if global_step % cfg.train.log_every == 0:
                elapsed = time.time() - t0
                lr      = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch:3d} | step {global_step:6d} | "
                    f"loss {loss.item():.4f} | lr {lr:.2e} | "
                    f"{elapsed:.0f}s"
                )
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LR", lr, global_step)
                writer.add_scalar("Epoch", epoch, global_step)

            # Validate + checkpoint
            if global_step % cfg.train.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  → val loss: {val_loss:.4f}")
                writer.add_scalar("Loss/val", val_loss, global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    _save(model, cfg, global_step, val_loss, name="best_model")
                    print(f"  → best model saved ✓")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  → early stopping (no improvement for {patience} evals)")
                        _save(model, cfg, global_step, val_loss, name="final")
                        writer.close()
                        print(f"Best val loss: {best_val_loss:.4f}")
                        print(f"Total time:    {(time.time()-t0)/60:.1f} min")
                        print(f"\nChat with Mercy:\n  python -m tlha chat")
                        return

            if global_step % cfg.train.save_every == 0:
                _save(model, cfg, global_step, loss.item(),
                      name=f"step_{global_step:06d}")

    # Final
    val_loss = evaluate(model, val_loader, device)
    _save(model, cfg, global_step, val_loss, name="final")
    writer.add_scalar("Loss/val", val_loss, global_step)
    writer.close()
    print("─" * 60)
    print(f"Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total time:    {(time.time()-t0)/60:.1f} min")
    print(f"\nView training graphs:")
    print(f"  tensorboard --logdir logs")
    print(f"  → open http://localhost:6006")
    print(f"\nChat with Mercy:")
    print(f"  python -m tlha chat")


def _save(model, cfg, step, loss, name):
    path = os.path.join(cfg.train.checkpoint_dir, f"{name}.pt")
    torch.save({
        "step":             step,
        "loss":             loss,
        "model_state_dict": model.state_dict(),
        "config":           cfg,
        "vocab_size":       cfg.model.vocab_size,
    }, path)
    # Keep only last 3 step checkpoints to avoid disk bloat
    if name.startswith("step_"):
        import glob
        step_files = sorted(glob.glob(
            os.path.join(cfg.train.checkpoint_dir, "step_*.pt")
        ))
        for old_f in step_files[:-3]:
            os.remove(old_f)


def load_checkpoint(path: str, device: str = "cpu"):
    """Load checkpoint → (model, cfg). Works on any device."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    if "vocab_size" in ckpt:
        cfg.model.vocab_size = ckpt["vocab_size"]
        cfg.model.__post_init__()
    cfg.train.device = device
    model = MercyLLM(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


if __name__ == "__main__":
    cfg = TLHAConfig()
    train(cfg)
