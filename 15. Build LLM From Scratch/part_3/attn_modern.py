"""3.5 Modern attention — RoPE + GQA + sliding window + sink + KV cache.

What this file does
-------------------
The "kitchen sink" causal self-attention used by modern LLMs (Llama,
Mistral, Phi). It composes five upgrades over the Part 1 vanilla MHA:

  1) RoPE              — rotary position embedding inside Q and K
  2) GQA               — Q has n_head heads, K/V share fewer n_kv_head heads
  3) Sliding window    — restrict K/V to the last `sliding_window` tokens
  4) Attention sink    — always keep first `attention_sink` tokens
  5) Optional KV cache — concat past K/V from a passed-in cache (generation)

Where this fits in the modern Transformer block
-----------------------------------------------
    [ Input tokens (B, T, d_model)        ]
              |
    [ 3.1 RMSNorm 1                       ]
              |
    [ 3.5 Modern Attention                ]   <-- THIS FILE
        - Wq, Wk, Wv  (Wk/Wv smaller under GQA)
        - apply RoPE to Q, K          (3.2)
        - concat with KV cache         (3.4)
        - crop [sink || last window]   (3.4)
        - repeat_interleave K/V to H heads
        - F.scaled_dot_product_attention(causal=True)
        - merge heads, output proj
              |
    [ + residual                          ]
              |
    [ 3.1 RMSNorm 2  -> 3.3 SwiGLU FFN -> + residual -> output ]

GQA in one sentence
-------------------
    n_head     = 8     (queries: 8 separate "lookup vectors" per token)
    n_kv_head  = 2     (keys/values: only 2 distinct K and V vectors,
                        each shared by group_size = n_head/n_kv_head = 4
                        consecutive query heads)

The KV cache is the *bandwidth* bottleneck during inference (you re-load it
from HBM for every new token). Cutting the K/V tensors by 4x cuts the
memory bandwidth by ~4x, with negligible quality loss in practice.

Shapes (toy: B=1, T=4, d_model=12, n_head=4, n_kv_head=2, d_head=3)
-------------------------------------------------------------------
  input x          : (1, 4, 12)           = (B, T, d_model)
  Wq(x)            : (1, 4, 12)           = (B, T, n_head * d_head)
  Wk(x), Wv(x)     : (1, 4, 6)  each      = (B, T, n_kv_head * d_head)
  q view + transp  : (1, 4, 4, 3)         = (B, n_head,    T, d_head)
  k,v view + transp: (1, 2, 4, 3)         = (B, n_kv_head, T, d_head)
  q,k after RoPE   : same shapes
  with KV cache    : (1, 2, T_past+T, 3)   for k_all, v_all
  after sink+win   : (1, 2, sink+window, 3) at most
  expand to n_head : (1, 4, *, 3)         = repeat_interleave(group_size=2)
  sdpa output      : (1, 4, T, 3)
  merge heads      : (1, T, 12)           = (B, T, d_model)
  output proj      : (1, T, 12)

Math (one new query token, with cache, RoPE, GQA expand)
--------------------------------------------------------
    q  = RoPE_pos(  W_q  x )                    # (B, H,  T, D)
    k  = RoPE_pos(  W_k  x )                    # (B, Hk, T, D)
    v  =           W_v  x                       # (B, Hk, T, D)
    k_all, v_all = cat([cache, k|v], dim=T)
    crop to [sink || last window]
    expand K,V from Hk to H (each K row repeated group_size times)
    scores = q @ k_all^T / sqrt(D)              # (B, H, T, T_all)
    weights = softmax( scores + causal_mask )
    out     = weights @ v_all                   # (B, H, T, D)
    return  out @ W_o, new_cache

Visualization
-------------
See notebook section 3.5:
  * GQA grouping diagram (4 query heads -> 2 KV heads)
  * sliding-window + sink mask on a long T (visual diagonal band + leftmost cols)
  * cache growth timeline (length over generation steps)

Parameter count (no biases)
---------------------------
  W_q : d_model * (n_head    * d_head) = d_model^2
  W_k : d_model * (n_kv_head * d_head) = d_model^2 / group_size
  W_v : d_model * (n_kv_head * d_head) = d_model^2 / group_size
  W_o : d_model * d_model              = d_model^2

  total: 2 * d_model^2 + 2 * d_model^2 / group_size
      vs full MHA: 4 * d_model^2
      savings:    2 * d_model^2 * (1 - 1/group_size)   for K and V
"""
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope_custom import RoPECache, apply_rope_single
from kv_cache import KVCache  # your existing class

class CausalSelfAttentionModern(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None):  # ← NEW
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head      # ← NEW (GQA defaults to MHA)
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        self.group_size = self.n_head // self.n_kv_head
        self.d_head = n_embd // n_head

        # Separate projections for Q vs K/V (sizes differ under GQA)  ← CHANGED
        self.wq  = nn.Linear(n_embd, self.n_head   * self.d_head, bias=False)
        self.wk  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.use_rope = rope
        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink

    def _maybe_init_rope(self, device):
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """x: (B,T,C). If kv_cache given, we assume generation (T small, often 1)."""
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        # Projections
        q = self.wq(x).view(B, T, self.n_head,   self.d_head).transpose(1, 2)    # (B,H, T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)

        # RoPE on *current* tokens (cached keys are already rotated)
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rope_single(q, cos, sin)   # (B,H, T,D)
            k = apply_rope_single(k, cos, sin)   # (B,Hk,T,D)

        # Concatenate past cache (cache is stored in Hk heads)
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Sliding-window + attention-sink (crop along seq length)
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # --- GQA expand: repeat K/V heads to match Q heads before attention ---
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
        else:
            k_attn, v_attn = k_all, v_all

        # Scaled dot-product attention (PyTorch scales internally)
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                           attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_causal)          # (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # Update KV cache (store compact Hk heads, not expanded)
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk,*,D)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = KVCache(k_new, v_new)
        return y, new_cache
