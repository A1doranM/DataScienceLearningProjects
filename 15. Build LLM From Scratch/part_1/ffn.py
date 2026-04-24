"""1.5 Feed-Forward Network (FFN / position-wise MLP).

What this file does
-------------------
Applies a small two-layer MLP independently to every token position. The
first Linear expands to a wider hidden dim (mult * d_model, typically 4x),
GELU applies a smooth non-linearity, the second Linear projects back down
to d_model. No information is mixed across positions — that's attention's
job.

Think of it as: attention = communication, FFN = computation.

Where this fits in the Transformer block
----------------------------------------
    [ Input tokens (B, T, d_model) ]
              |
    [ 1.1 Positional Encoding      ]
              |
    [ 1.5 LayerNorm 1              ]
              |
    [ 1.3/1.4 Multi-Head Attention ]
              |
    [ + residual                   ]
              |
    [ 1.5 LayerNorm 2              ]
              |
    [ 1.5 Feed-Forward             ]   <-- THIS FILE
              |
    [ + residual                   ]
              |
    [ Block output (B, T, d_model) ]

Math
----
    h   = x @ W_1 + b_1            # expand:  d_model    -> 4*d_model
    h   = GELU(h)                  # smooth non-linearity
    out = h @ W_2 + b_2            # project: 4*d_model  -> d_model
    out = Dropout(out)

GELU (Gaussian Error Linear Unit):
    GELU(x) = x * Phi(x)        where Phi is the standard normal CDF
  Behaves like ReLU for x >> 0, ~0 for x << 0, smoothly curved around 0
  (no sharp kink). Its continuous derivative helps gradient flow.

Why expand 4x?
--------------
The expansion gives the network more representational capacity to learn
complex non-linear transformations. The FFN typically holds ~2/3 of all
parameters in a Transformer — it is where most of the "thinking" happens.

Visualization
-------------
See notebook section 1.5a — GELU vs ReLU plotted on the same axes to show
the smooth-vs-kinked difference near zero.

Shapes
------
  input     : (B, T, d_model)
  hidden    : (B, T, mult * d_model)
  output    : (B, T, d_model)     — same shape as input

Parameter count (mult=4)
------------------------
  W_1, b_1 : d_model * 4*d_model + 4*d_model
  W_2, b_2 : 4*d_model * d_model + d_model
  total    : ~8 * d_model^2   — dominant parameter cost of the block.
"""

import torch.nn as nn

class FeedForward(nn.Module):
    """1.5 FFN with expansion factor `mult`.

    Dimensions:
      input:     (B, T, d_model)
      inner:     (B, T, mult*d_model)
      output:    (B, T, d_model)

    `mult*d_model` means the hidden width is `mult` times larger than `d_model`.
    Typical values: mult=4 for GELU FFN in GPT-style blocks.
    """
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)