"""
CLI for Mercy: The Only Human Left with Pigeon Gerald.

Commands:
    python -m tlha prepare              generate data + train tokenizer
    python -m tlha train                train the model
    python -m tlha chat                 chat with Mercy (auto-loads best checkpoint)
    python -m tlha chat --prompt "..."  single prompt, non-interactive
    python -m tlha chat --model path    use a specific checkpoint
    python -m tlha chat --temperature   sampling temperature (default 0.75)
    python -m tlha eval                 run held-out evaluation cases
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="tlha",
        description="Mercy: The Only Human Left with Pigeon Gerald",
    )
    sub = parser.add_subparsers(dest="command")

    # ── prepare ───────────────────────────────────────────────────────────────
    sub.add_parser(
        "prepare",
        help="Generate 80K conversations, train tokenizer, encode tensors",
    )

    # ── train ─────────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs (default: 50)"
    )
    train_p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (default: 128)"
    )
    train_p.add_argument(
        "--device", default="auto",
        help="Device: auto | mps | cuda | cpu  (default: auto)"
    )

    # ── chat ──────────────────────────────────────────────────────────────────
    chat_p = sub.add_parser("chat", help="Chat with Mercy")
    chat_p.add_argument("--model",       default=None,
                        help="Path to checkpoint .pt file")
    chat_p.add_argument("--prompt",      default=None,
                        help="Single prompt (non-interactive mode)")
    chat_p.add_argument("--temperature", type=float, default=0.75,
                        help="Sampling temperature (default: 0.75)")
    chat_p.add_argument("--device",      default="auto",
                        help="Device: auto | mps | cuda | cpu")

    # ── eval ──────────────────────────────────────────────────────────────────
    sub.add_parser("eval", help="Run held-out evaluation cases")

    args = parser.parse_args()

    # ── dispatch ──────────────────────────────────────────────────────────────

    if args.command == "prepare":
        from .generate_data import generate_dataset, save_dataset
        from .prepare_data import prepare
        from .config import TLHAConfig

        cfg = TLHAConfig()
        print("Generating 100,000 conversations across 143 topics...")
        records = generate_dataset(80000)
        save_dataset(records, cfg.train.data_dir)

        print("\nTraining tokenizer + encoding tensors...")
        prepare(cfg)
        print("\nAll done. Run:  python -m tlha train")

    elif args.command == "train":
        from .train import train
        from .config import TLHAConfig

        cfg = TLHAConfig()
        if args.epochs:
            cfg.train.num_epochs = args.epochs
        if args.batch_size:
            cfg.train.batch_size = args.batch_size
        if args.device:
            cfg.train.device = args.device
        train(cfg)

    elif args.command == "chat":
        from .inference import run
        run(
            checkpoint=args.model,
            prompt=args.prompt,
            temperature=args.temperature,
            device=args.device,
        )

    elif args.command == "eval":
        import os
        from .config import resolve_device
        from .inference import load_model_local
        from .eval_cases import run_eval

        device    = resolve_device("auto")
        ckpt_path = os.path.join("checkpoints", "best_model.pt")
        tok_path  = os.path.join("data", "tokenizer.json")

        if not os.path.exists(ckpt_path):
            print("No checkpoint found. Train first:  python -m tlha train")
            sys.exit(1)

        model, tokenizer, _ = load_model_local(ckpt_path, tok_path, device)
        run_eval(model, tokenizer, device)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
