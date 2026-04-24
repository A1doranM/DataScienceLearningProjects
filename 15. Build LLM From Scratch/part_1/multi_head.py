"""1.4 Multi-head self-attention with explicit shape tracing.

What this file does
-------------------
Runs n_head independent attention heads in parallel over the same input, each
operating in a d_head = d_model / n_head subspace, then concatenates their
outputs and applies a final learned projection. A single combined W_qkv of
shape (d_model, 3*d_model) replaces three separate matrices — identical math,
faster on GPU.

Where this fits in the Transformer block
----------------------------------------
  [ Input tokens (B, T, d_model) ]
                 |
  [ 1.1 Positional Encoding      ]
                 |
  [ 1.5 LayerNorm 1              ]
                 |
==> 1.3/1.4 Multi-Head Attention ]
                 |
  [ + residual                   ]
                 |
  [ 1.5 LayerNorm 2              ]
                 |
  [ 1.5 Feed-Forward             ]
                 |
  [ + residual                   ]
                 |
  [ Block output (B, T, d_model) ]

Why multiple heads?
-------------------
A single attention head can only learn ONE pattern of "who should attend to
whom". But language has many patterns happening simultaneously:
  * head A might learn "attend to the previous word"   (local context)
  * head B might learn "attend to the subject"          (syntactic)
  * head C might learn "attend to punctuation"          (structural)
Multiple heads let the model learn these in parallel.

Math
----
    qkv = x @ W_qkv                                      # (B, T, 3*d_model)
    split into q, k, v    each reshaped to              # (B, heads, T, d_head)
    attn  = q @ k^T / sqrt(d_head)                      # (B, heads, T, T)
    attn  = attn.masked_fill(causal_mask, -infinity)
    w     = softmax(attn, dim=-1)
    ctx   = w @ v                                       # (B, heads, T, d_head)
    out   = concat heads (transpose+view)  @  W_o        # (B, T, d_model)

Visualization
-------------
See notebook section 1.4:
  * per-head attention heatmaps in a grid (from demo_visualize_multi_head.py)
  * "split heads" visual: one big vector -> rows of colored slices per head

Shapes (d_model=64, n_head=4, d_head=16, T=10)
----------------------------------------------
  input x             : (B, 10, 64)
  after W_qkv         : (B, 10, 192)
  view (B,T,3,H,d_h)  : (B, 10, 3, 4, 16)
  q, k, v (unbind)    : each (B, 10, 4, 16)
  transpose(1,2)      : each (B, 4, 10, 16)
  scores q@k^T/sqrt   : (B, 4, 10, 10)
  weights softmax     : (B, 4, 10, 10)
  ctx = w @ v         : (B, 4, 10, 16)
  merge back          : (B, 10, 64)
  out = merge @ W_o   : (B, 10, 64)

Parameter count
---------------
  W_qkv : d_model * (3 * d_model)   = 3 * d_model^2
  W_o   : d_model * d_model         =     d_model^2
  total : 4 * d_model^2             — the attention parameter budget.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class MultiHeadSelfAttention(nn.Module):
    """1.4 Multi-head attention with explicit shape tracing.

    Dimensions (before masking):
      x:      (B, T, d_model)
      qkv:    (B, T, 3*d_model)
      view→   (B, T, 3, n_head, d_head)   where d_head = d_model // n_head
      split→  q,k,v each (B, T, n_head, d_head)
      swap→   (B, n_head, T, d_head)
      scores: (B, n_head, T, T) = q @ k^T / sqrt(d_head)
      weights:(B, n_head, T, T) = softmax(scores)
      ctx:    (B, n_head, T, d_head) = weights @ v
      merge:  (B, T, n_head*d_head) = (B, T, d_model)
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # (B,T,d_model)
        B, T, C = x.shape
        qkv = self.qkv(x)                          # (B,T,3*C)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)  # (B,T,3,heads,dim)
        if self.trace_shapes:
            print("qkv view:", qkv.shape)
        q, k, v = qkv.unbind(dim=2)               # each: (B,T,heads,dim)
        q = q.transpose(1, 2)                      # (B,heads,T,dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.trace_shapes:
            print("q:", q.shape, "k:", k.shape, "v:", v.shape)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,heads,T,T)
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask, float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        ctx = torch.matmul(w, v)                  # (B,heads,T,dim)
        if self.trace_shapes:
            print("weights:", w.shape, "ctx:", ctx.shape)
        out = ctx.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,d_model)
        out = self.proj(out)
        if self.trace_shapes:
            print("out:", out.shape)
        return out, w