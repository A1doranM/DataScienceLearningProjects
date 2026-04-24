"""Causal mask helper for autoregressive self-attention.

What this file does
-------------------
Builds the upper-triangular boolean mask that prevents token at position t
from attending to tokens at positions t+1, t+2, ... (the future). This is
the "causal" (a.k.a. decoder-only, autoregressive) pattern used by GPT-style
language models — every token may look at itself and earlier tokens only.

Where this fits in the Transformer block
----------------------------------------
The mask is applied inside the Multi-Head Attention block, after computing
raw scores and before softmax.

    [ Input tokens (B, T, d_model) ]
              |
    [ 1.1 Positional Encoding      ]
              |
    [ 1.5 LayerNorm 1              ]
              |
    [ 1.3/1.4 Multi-Head Attention ]   <-- THIS FILE
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

Math
----
Given sequence length T, the mask M is the (T, T) matrix with

    M[i, j] = True   if j > i    (future — blocked)
    M[i, j] = False  if j <= i   (past or self — allowed)

Applied to the attention scores S before softmax:

    S[i, j] = -infinity   where M[i, j] is True
    W = softmax(S)          — the -inf positions become exactly 0

For T=5 the mask looks like (X = blocked, . = allowed):
    .  X  X  X  X
    .  .  X  X  X
    .  .  .  X  X
    .  .  .  .  X
    .  .  .  .  .

Visualization
-------------
See notebook section 1.3 — the mask is shown as a red/white heatmap with
X and . markers on each cell.

Shapes
------
  return   : (1, 1, T, T)   — broadcasts with (B, heads, T, T) attention scores.
"""

import torch

def causal_mask(T: int, device=None):
    """Returns a bool mask where True means *masked* (disallowed).
    Shape: (1, 1, T, T) suitable for broadcasting with (B, heads, T, T).
    """
    m = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    return m.view(1, 1, T, T)