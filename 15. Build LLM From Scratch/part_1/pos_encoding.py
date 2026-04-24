"""1.1 Positional Encoding — learned and sinusoidal variants.

What this file does
-------------------
Implements two ways to inject position information into token embeddings:
  * LearnedPositionalEncoding    — trainable nn.Embedding(max_len, d_model) table
  * SinusoidalPositionalEncoding — fixed sin/cos waves at geometric frequencies

Self-attention is permutation-invariant (it sees tokens as a *set*), so the
model cannot distinguish "cat sat" from "sat cat" without extra help. The
positional encoding is that extra help: a vector added to each token embedding
that encodes its position index t = 0, 1, 2, ....

Where this fits in the Transformer block
----------------------------------------
  [ Input tokens (B, T, d_model) ]
                 |
==> 1.1 Positional Encoding      ]
                 |
  [ 1.5 LayerNorm 1              ]
                 |
  [ 1.3/1.4 Multi-Head Attention ]
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

Math (sinusoidal variant)
-------------------------
For position `pos` and dimension index `i`:

    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

  * Even-indexed dims use sine, odd-indexed dims use cosine.
  * The denominator 10000^(2i/d_model) gives each dim its own wavelength:
    low-index dims wiggle fast (sensitive to small position differences),
    high-index dims wiggle slowly (sensitive to long-range differences).
  * Property: dot(PE(pos), PE(pos+k)) depends only on k — the model can
    reason about *relative* offsets.

Visualization
-------------
In the walkthrough notebook (part_1_walkthrough.ipynb, section 1.1):
  * Heatmap of the (max_len, d_model) PE matrix — the classic striped pattern.
  * Line plot of dims 0/4/16/32 — showing frequency decreases with dim index.

Shapes
------
  input  x      : (B, T, d_model)
  PE table      : (max_len, d_model)
  output x + pe : (B, T, d_model)   — PE is added, not concatenated.
"""

import math
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model) — we only need its T and device
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        pos_emb = self.emb(pos)  # (T, d_model)
        return x + pos_emb.unsqueeze(0)  # broadcast over batch

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        return x + self.pe[:T].unsqueeze(0)