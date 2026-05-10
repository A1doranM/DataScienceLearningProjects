"""3.6 Modern Transformer block — RMSNorm + Modern Attention + SwiGLU.

What this file does
-------------------
Composes the three modern subcomponents (3.1, 3.5, 3.3) into a single
repeatable block, with the same pre-norm / residual pattern as Part 1 but
each ingredient swapped:

    Part 1 vanilla:   LayerNorm  -> MHA        -> +  -> LayerNorm  -> GELU FFN  -> +
    Part 3 modern:    RMSNorm    -> ModernAttn -> +  -> RMSNorm    -> SwiGLU     -> +

Feature flags allow ablations (RMSNorm <-> LayerNorm, SwiGLU <-> GELU)
without rewriting code — useful for the demos.

Where this fits in the modern Transformer block
-----------------------------------------------
    [ Input tokens (B, T, d_model)        ]   <-- THIS FILE (whole block)
              |
    [ 3.1 RMSNorm 1                       ]   <-- THIS FILE
              |
    [ 3.5 Modern Attention                ]   <-- THIS FILE
              |
    [ + residual                          ]   <-- THIS FILE
              |
    [ 3.1 RMSNorm 2                       ]   <-- THIS FILE
              |
    [ 3.3 SwiGLU FFN                      ]   <-- THIS FILE
              |
    [ + residual                          ]   <-- THIS FILE
              |
    [ Block output (B, T, d_model)        ]   <-- THIS FILE

The forward pass in two lines (with KV cache)
---------------------------------------------
    a, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
    x = x + a                                # attention residual
    x = x + self.ffn(self.ln2(x))            # FFN residual
    return x, kv_cache

Why pre-norm (still)?
---------------------
Same reason as Part 1: the residual highway stays "clean" — raw x flows
through the block with only added contributions, never normalized away.
Pre-norm + RMSNorm is the standard combo since GPT-NeoX and Llama.

Shapes
------
  input x  : (B, T, d_model)
  output x : (B, T, d_model)
  kv_cache : KVCache(k=(B, Hk, T_total, D), v=(B, Hk, T_total, D)) or None

Visualization
-------------
See notebook section 3.6 — full shape trace through one block forward
pass, plus parameter count broken down by submodule.
"""
import torch.nn as nn
from rmsnorm import RMSNorm
from swiglu import SwiGLU
from attn_modern import CausalSelfAttentionModern

class TransformerBlockModern(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln1 = Norm(n_embd)
        self.attn = CausalSelfAttentionModern(n_embd, n_head, dropout, rope, max_pos, sliding_window, attention_sink, n_kv_head)
        self.ln2 = Norm(n_embd)
        self.ffn = SwiGLU(n_embd, mult=4, dropout=dropout) if use_swiglu else nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
        )
    def forward(self, x, kv_cache=None, start_pos: int = 0):
        a, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, kv_cache