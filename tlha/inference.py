"""
Inference — chat with Mercy, mercy: the only human left with pigeon gerald.

Loads trained model and runs an interactive terminal loop.
Mercy speaks in short, lowercase sentences. She has been
alone for 847 days. She has a pigeon named Gerald.

Usage:
    python -m tlha chat
    python -m tlha chat --prompt "tell me a joke"
    python -m tlha chat --model checkpoints/best_model.pt
    python -m tlha chat --temperature 0.9
"""

import os
import torch
from tokenizers import Tokenizer

from .config import TLHAConfig, resolve_device
from .model import MercyLLM
from .prepare_data import USER_TOKEN, CHAR_TOKEN, END_TOKEN

HF_REPO = "your-username/mercy-15M"


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_local(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str,
) -> tuple["MercyLLM", "Tokenizer", "TLHAConfig"]:
    """Load model + tokenizer from local files."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    # Restore actual vocab size from checkpoint (never trust default 4096)
    if "vocab_size" in ckpt:
        cfg.model.vocab_size = ckpt["vocab_size"]
        cfg.model.__post_init__()

    cfg.train.device = device
    model = MercyLLM(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, cfg


def load_model_hf(repo: str, device: str):
    """Download model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id=repo, filename="model.pt")
    tok_path  = hf_hub_download(repo_id=repo, filename="tokenizer.json")
    return load_model_local(ckpt_path, tok_path, device)


# ── response generation ───────────────────────────────────────────────────────

def respond(
    prompt: str,
    model: MercyLLM,
    tokenizer: Tokenizer,
    device: str,
    temperature: float = 0.75,
    top_k: int = 40,
    max_new_tokens: int = 80,
) -> str:
    """
    Generate one response from Mercy given a user prompt.
    Format: [USER] prompt [MERCY] → generate until [END] or max_new_tokens.
    """
    user_id = tokenizer.token_to_id(USER_TOKEN)
    char_id = tokenizer.token_to_id(CHAR_TOKEN)
    end_id  = tokenizer.token_to_id(END_TOKEN)

    inp_ids    = tokenizer.encode(prompt.strip()).ids
    prompt_ids = [user_id] + inp_ids + [char_id]
    input_t    = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_t,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    generated = output_ids[0, len(prompt_ids):].tolist()

    # Trim at [END]
    if end_id in generated:
        generated = generated[:generated.index(end_id)]

    decoded = tokenizer.decode(generated)
    # Remove BPE space prefix characters
    decoded = decoded.replace("ġ", " ").replace("Ġ", " ")
    # Fix spaces before punctuation: mercy 's -> mercy's, doesn 't -> doesn't
    import re
    decoded = re.sub(r" ([',\.!?;:])", r"\1", decoded)
    decoded = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", decoded)
    # Fix tokenization splits in proper nouns and roads
    decoded = re.sub(r"\b(m|a|b|f) (\d+)\b", r"\1\2", decoded)  # m7, a1, b2
    decoded = re.sub(r"\b(day) (\d+)\b", r"\1 \2", decoded)      # keep "day 847"
    # Fix digit splits: "103 5" → "1035", "day 3 5 1" → "day 351"
    # Safe: only rejoins sequences of digits separated by single spaces
    decoded = re.sub(r'(\d+)(?: (\d+))+', lambda m: re.sub(r' ', '', m.group(0)), decoded)
    # Fix known BPE word splits — safe dictionary (no retraining needed)
    _WORD_FIXES = [
        ("phot o al b um", "photo album"), ("al b um", "album"), ("phot o", "photo"),
        ("sme lled", "smelled"), ("so ap", "soap"), ("m r chen", "mr chen"),
        ("b ri s b ane", "brisbane"), ("os ak a", "osaka"), ("it al y", "italy"),
        ("s al tin es", "saltines"), ("sal tin es", "saltines"),
        ("pe aches", "peaches"), ("pe anut", "peanut"), ("gr anola", "granola"),
        ("ex pired", "expired"), ("p owdered", "powdered"), ("o ats", "oats"),
        ("half - dr un k", "half-drunk"), ("dr un k", "drunk"),
        ("ta b les", "tables"), ("tab les", "tables"),
        ("wareh ouse", "warehouse"), ("ware house", "warehouse"),
        ("dist ri ct", "district"), ("dist rict", "district"),
        ("respon s ible", "responsible"), ("cer ta int y", "certainty"),
        ("terri fied", "terrified"), ("dec ent", "decent"), ("end less", "endless"),
        ("sur p rise", "surprise"), ("ed ible", "edible"),
        ("tal ked", "talked"), ("a im lessly", "aimlessly"),
        ("acc ess", "access"), ("per mission", "permission"),
        ("co l ours", "colours"), ("yes ter day", "yesterday"),
        ("f am ili ar", "familiar"), ("cl aim ed", "claimed"),
        ("tw isted", "twisted"), ("as se mb led", "assembled"),
        ("la bour", "labour"), ("the ore tical", "theoretical"),
        ("bl ood", "blood"), ("vi ol ence", "violence"),
        ("ex cept", "except"), ("com fort", "comfort"),
        ("r ating", "rating"), ("char ges", "charges"),
        ("system atic", "systematic"), ("meth o do logy", "methodology"),
        ("sha pes", "shapes"), ("cat ches", "catches"), ("tast ed", "tasted"),
        ("ru gby", "rugby"), ("lea gu e", "league"), ("arg ument", "argument"),
        ("reci pe", "recipe"), ("observ ations", "observations"),
        ("le ather", "leather"), ("child ren", "children"),
        ("d v ds", "dvds"), ("mid - st ep", "mid-step"),
        ("b ir th day c ard ad d ressed", "birthday card addressed"),
        ("b ir th day c ard", "birthday card"),
        ("b ir th day", "birthday"), ("c ard ad d ressed", "card addressed"),
        ("every where", "everywhere"), ("some thing", "something"),
        ("every thing", "everything"), ("any thing", "anything"),
        ("every one", "everyone"), ("some one", "someone"),
        ("struct ured", "structured"), ("struct ure", "structure"),
        ("infrastr ucture", "infrastructure"), ("man aging", "managing"),
        ("out side", "outside"), ("break down", "breakdown"),
        ("break fast", "breakfast"), ("sun rise", "sunrise"),
        ("sun set", "sunset"), ("over all", "overall"),
        ("mean while", "meanwhile"),
    ]
    for wrong, right in _WORD_FIXES:
        decoded = decoded.replace(wrong, right)
    return " ".join(decoded.split()).strip().lower()


# ── chat loop ─────────────────────────────────────────────────────────────────

def _banner():
    print()
    print("  Mercy: The Only Human Left with Pigeon Gerald")
    print("  day 847. you found her.")
    print("  type 'quit' to leave.\n")
    print("  " + "─" * 40)
    print()


def chat_loop(
    model: MercyLLM,
    tokenizer: Tokenizer,
    device: str,
    single_prompt: str = None,
    temperature: float = 0.75,
):
    if single_prompt:
        reply = respond(single_prompt, model, tokenizer, device, temperature)
        print(f"You> {single_prompt}")
        print(f"Mercy> {reply}\n")
        return

    _banner()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            reply = respond("goodbye", model, tokenizer, device, temperature)
            print(f"\nMercy> {reply}\n")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q", "bye", "goodbye"}:
            reply = respond("goodbye", model, tokenizer, device, temperature)
            print(f"Mercy> {reply}\n")
            break

        reply = respond(user_input, model, tokenizer, device, temperature)
        print(f"Mercy> {reply}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def run(
    checkpoint: str = None,
    prompt: str = None,
    temperature: float = 0.75,
    device: str = "auto",
):
    device = resolve_device(device)

    if checkpoint:
        # Explicit checkpoint path
        tok_path = os.path.join(
            os.path.dirname(checkpoint), "..", "data", "tokenizer.json"
        )
        if not os.path.exists(tok_path):
            tok_path = os.path.join(
                os.path.dirname(checkpoint), "tokenizer.json"
            )
        model, tokenizer, _ = load_model_local(checkpoint, tok_path, device)

    else:
        # Auto-discover: local checkpoints first, then HuggingFace
        local_ckpt = os.path.join("checkpoints", "best_model.pt")
        local_tok  = os.path.join("data", "tokenizer.json")

        if os.path.exists(local_ckpt) and os.path.exists(local_tok):
            print(f"Loading local checkpoint: {local_ckpt}")
            model, tokenizer, _ = load_model_local(local_ckpt, local_tok, device)
        else:
            print(f"Downloading from HuggingFace: {HF_REPO}")
            model, tokenizer, _ = load_model_hf(HF_REPO, device)

    chat_loop(model, tokenizer, device, single_prompt=prompt,
              temperature=temperature)
