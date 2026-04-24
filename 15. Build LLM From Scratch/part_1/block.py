"""1.6 Transformer block — LN -> MHA -> + -> LN -> FFN -> +.

What this file does
-------------------
Composes LayerNorm, Multi-Head Attention, LayerNorm, and FeedForward with
two residual skip connections into a single repeatable unit. A full GPT
stacks N of these blocks (N = 6 / 12 / 24 / 96 depending on model size).

Where this fits in the Transformer block
----------------------------------------
The whole block — every node is active here:

==> Input tokens (B, T, d_model) ]
                 |
==> 1.1 Positional Encoding      ]
                 |
==> 1.5 LayerNorm 1              ]
                 |
==> 1.3/1.4 Multi-Head Attention ]
                 |
==> + residual                   ]
                 |
==> 1.5 LayerNorm 2              ]
                 |
==> 1.5 Feed-Forward             ]
                 |
==> + residual                   ]
                 |
==> Block output (B, T, d_model) ]

The forward pass in two lines
-----------------------------
    x = x + self.attn(self.ln1(x))[0]      # communication (tokens exchange info)
    x = x + self.ffn (self.ln2(x))         # computation   (each token processes)

Math — Pre-Norm pattern
-----------------------
    x_1 = x    + MHA(LayerNorm(x))
    x_2 = x_1  + FFN(LayerNorm(x_1))
    output = x_2

Two crucial design choices:

  1. Pre-Norm: LayerNorm is applied BEFORE each sublayer, not after. Makes
     training much more stable than the original Post-Norm paper. The
     residual path stays "clean" — raw x is added without normalization
     in between.

  2. Residual connections: `x + sublayer(...)` provides a gradient highway.
     Even if the sublayer attenuates gradients heavily, the direct skip
     keeps the signal alive — essential for training deep stacks.

LayerNorm
---------
Normalizes each token independently (across its d_model features) to zero
mean and unit variance, then scales and shifts by learnable gamma, beta:

    LN(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
    where mu, sigma^2 are computed across the last (feature) dimension.

Unlike BatchNorm, it doesn't depend on batch statistics — same behavior
at train and inference time.

Visualization
-------------
See notebook section 1.6 — full shape trace line by line through one
forward pass, plus parameter count broken down by submodule.

Shapes
------
  input x  : (B, T, d_model)
  output   : (B, T, d_model)     — block preserves shape (so it can be stacked)
"""

import torch.nn as nn
from multi_head import MultiHeadSelfAttention
from ffn import FeedForward

class TransformerBlock(nn.Module):
    """1.6 Transformer block = LN → MHA → residual → LN → FFN → residual."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mult=4, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x