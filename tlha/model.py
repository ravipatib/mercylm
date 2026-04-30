"""
Mercy — transformer model (Mercy: The Only Human Left with Pigeon Gerald).

A small, readable vanilla transformer. No GQA, no RoPE, no SwiGLU.
Standard multi-head attention + ReLU feedforward + LayerNorm.
Learned positional embeddings. Weight-tied input/output embeddings.

~15M parameters at the default config. Trains in ~10 min on MPS, ~8 min on Colab T4
on any modern GPU, including Apple Silicon via MPS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product multi-head attention with causal mask.
    Each head independently attends over the sequence; outputs are
    concatenated and projected back to embed_dim.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = math.sqrt(cfg.head_dim)

        # Fused QKV projection — one matmul instead of three
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=False)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv(x)                                   # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)                      # each (B, T, C)

        # Reshape to (B, heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2, -1)) / self.scale     # (B, H, T, T)
        causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        # Weighted sum of values, merge heads
        out = (weights @ v)                                  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Two-layer feedforward block: Linear → ReLU → Linear.
    Expands to ffn_dim then projects back to embed_dim.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ffn_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention + residual → LayerNorm → FFN + residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MercyLLM(nn.Module):
    """
    Mercy — mercy: the only human left with pigeon gerald.

    Embedding → N × TransformerBlock → LayerNorm → LM head.
    Input and output embeddings share the same weight matrix.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.context_len, cfg.embed_dim)
        self.embed_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.embed_dim)

        # LM head — weight-tied with token embeddings
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            token_ids: (B, T) integer token ids
            targets:   (B, T) shifted token ids for loss computation, or None

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss if targets provided, else None
        """
        B, T = token_ids.shape
        assert T <= self.cfg.context_len, \
            f"Sequence length {T} exceeds context_len {self.cfg.context_len}"

        positions = torch.arange(T, device=token_ids.device)
        x = self.embed_drop(self.token_embed(token_ids) + self.pos_embed(positions))

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)                            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 60,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with temperature + top-k sampling.

        Args:
            prompt_ids:     (1, T) seed token ids
            max_new_tokens: maximum tokens to generate
            temperature:    > 1 = more random, < 1 = more focused
            top_k:          restrict sampling to top-k logits

        Returns:
            (1, T + max_new_tokens) full token sequence
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to context window
            ctx = ids[:, -self.cfg.context_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature         # (1, vocab_size)

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

        return ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
