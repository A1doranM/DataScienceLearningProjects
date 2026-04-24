"""1.3 Single-head self-attention as an nn.Module (PyTorch).

What this file does
-------------------
Same math as attn_numpy_demo.py, but wrapped in an nn.Module so the W_q, W_k,
W_v projection matrices are *learned* via backprop. Returns both the output
and the attention weights (so we can inspect what the head is doing).

Where this fits in the Transformer block
----------------------------------------
This IS the attention mechanism — still a single head; the multi-head version
that a real Transformer uses lives in multi_head.py.

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

Math — identical to section 1.2
-------------------------------
    q = x @ W_q       k = x @ W_k       v = x @ W_v       (nn.Linear w/o bias)
    attn = q @ k^T / sqrt(d_k)
    attn = attn.masked_fill(causal_mask, -infinity)       (hide future)
    w    = softmax(attn, dim=-1)
    out  = w @ v                                          (weighted values)

Three PyTorch-specific differences from the NumPy demo
------------------------------------------------------
  1. nn.Linear(d_model, d_k, bias=False) creates the three trainable matrices.
  2. masked_fill applies the causal mask cleanly in one call.
  3. Returns (output, attention_weights) — the weights are useful for viz.

Visualization
-------------
See notebook section 1.3 — the causal mask is shown as a red/white heatmap
with X / . markers for blocked / allowed positions.

Shapes
------
  input  x      : (B, T, d_model)
  q, k, v       : each (B, T, d_k)
  attn scores   : (B, T, T)
  weights       : (B, T, T)   — row-stochastic, lower-triangular
  output        : (B, T, d_k)

Parameter count
---------------
  3 * (d_model * d_k) — three weight matrices (no biases).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class SingleHeadSelfAttention(nn.Module):
    """1.3 Single-head attention (explicit shapes)."""
    def __init__(self, d_model: int, d_k: int, dropout: float = 0.0, trace_shapes: bool = False):
        super().__init__()
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # x: (B, T, d_model)
        B, T, _ = x.shape
        q = self.q(x)  # (B,T,d_k)
        k = self.k(x)  # (B,T,d_k)
        v = self.v(x)  # (B,T,d_k)
        if self.trace_shapes:
            print(f"q {q.shape}  k {k.shape}  v {v.shape}")
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,T,T)
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask.squeeze(1), float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        out = torch.matmul(w, v)  # (B,T,d_k)
        if self.trace_shapes:
            print(f"weights {w.shape}  out {out.shape}")
        return out, w