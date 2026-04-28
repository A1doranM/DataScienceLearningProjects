"""2.3 Tiny GPT -- Part 1 block becomes a trainable language model.

What this file does
-------------------
Defines a compact decoder-only GPT:
  * byte token embedding
  * learned positional embedding
  * repeated Transformer blocks
  * final LayerNorm
  * language-model head that predicts the next byte token

The attention and FFN pieces mirror Part 1, but here they are wired into a full
model that accepts integer token IDs and returns logits over the 256-byte
vocabulary.

Where this fits in the Part 2 training pipeline
-----------------------------------------------
    [ x: token IDs (B, T) ]
              |
    [ token + position embeddings ]     <-- THIS FILE
              |
    [ Transformer blocks from Part 1 ]   <-- THIS FILE
              |
    [ logits (B, T, vocab_size) ]        <-- THIS FILE
              |
    [ cross-entropy if targets exist ]
              |
    [ generate() for sampling ]

Connection to Part 1 and later parts
------------------------------------
Part 1 built the Transformer block from positional information, attention,
FFN, LayerNorm, and residuals. Part 2 wraps that block in embeddings, a final
prediction head, loss, and generation. Part 3 modernizes these internals with
RMSNorm, RoPE, SwiGLU, KV cache, and sliding windows. Parts 4-9 keep the same
high-level LM interface while changing scale, data, and objectives.

Math
----
    pos = [0, 1, ..., T-1]
    x = tok_emb(idx) + pos_emb(pos)
    x = Block_1(...Block_N(x)...)
    logits = x @ W_vocab

If targets are provided:
    loss = CE(logits.reshape(B*T, vocab_size), targets.reshape(B*T))

Sampling repeats:
    p(next_token | context) = softmax(filtered(logits_last / temperature))

Shapes
------
    idx:        LongTensor [B, T]
    embeddings: FloatTensor [B, T, n_embd]
    q,k,v:      FloatTensor [B, n_head, T, d_head]
    logits:     FloatTensor [B, T, vocab_size]
    loss:       scalar

Visualization
-------------
See notebook sections 2.3-2.5. They trace token IDs through embeddings,
attention blocks, logits, cross-entropy, and sampling filters.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Blocks (self-contained for isolation) ----
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_head = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):  # (B,T,C)
        B, T, C = x.shape
        # One projection produces [Q | K | V], then view exposes the head axis.
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # SDPA expects attention tensors as (B, heads, T, d_head).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_head)
        # PyTorch SDPA computes softmax(q @ k^T * scale) @ v with causal masking.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, mult=4, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ---- Tiny GPT ----
class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        # idx is (B, T); token and position tables lift it to (B, T, n_embd).
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Cross-entropy expects a flat list of B*T classification problems.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0,
                top_k: int | None = 50, top_p: float | None = None):
        from utils import top_k_top_p_filtering
        self.eval()
        # Guard: if the prompt is empty, start with a newline byte (10)
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            # Keep only the latest block_size tokens because the position table
            # is defined for contexts up to self.block_size.
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
