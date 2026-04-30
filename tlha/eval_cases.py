"""
Held-out evaluation cases for Mercy: The Only Human Left with Pigeon Gerald.

Run after training to quickly assess whether the character
sounds right. Not used during training — purely qualitative.

Usage:
    python -m tlha.eval_cases
"""

# These inputs are intentionally NOT in generate_data.py topics
# so they test generalisation, not memorisation.

EVAL_CASES = [
    # Existential
    "why do you keep going",
    "do you ever feel like giving up",
    "what's the hardest part",

    # Gerald
    "where's gerald right now",
    "does gerald understand you",
    "what would you do without gerald",

    # Unexpected / absurd
    "can you teach me something",
    "what's 2 + 2",
    "do you believe in ghosts",
    "have you tried meditation",
    "what's your favourite colour",

    # Human connection
    "i miss you already",
    "you're not alone anymore",
    "can i come find you",
    "i'm scared too",

    # Mundane
    "what did you have for breakfast",
    "do you exercise",
    "what time is it",
    "are you bored right now",

    # Deep
    "do you think you'll be remembered",
    "what was the last thing that made you laugh",
    "if you could send one message to the world, what would it say",
]


def run_eval(model, tokenizer, device, temperature=0.75):
    from .inference import respond
    print("\n── evaluation cases ──────────────────────────────────\n")
    for prompt in EVAL_CASES:
        reply = respond(prompt, model, tokenizer, device, temperature)
        print(f"You> {prompt}")
        print(f"Mercy> {reply}\n")


if __name__ == "__main__":
    import os
    from .config import resolve_device
    from .inference import load_model_local

    device = resolve_device("auto")
    ckpt = os.path.join("checkpoints", "best_model.pt")
    tok  = os.path.join("data", "tokenizer.json")

    if not os.path.exists(ckpt):
        print("No checkpoint found. Train first: python -m tlha train")
    else:
        model, tokenizer, _ = load_model_local(ckpt, tok, device)
        run_eval(model, tokenizer, device)
