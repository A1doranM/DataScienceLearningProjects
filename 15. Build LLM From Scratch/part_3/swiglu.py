"""3.3 SwiGLU — Gated FFN that replaces the GELU MLP (Llama / PaLM style).

What this file does
-------------------
Implements the SwiGLU feed-forward network used in modern LLMs. It splits
the input into two parallel projections, applies SiLU (Swish) to one of
them as a *gate*, multiplies them element-wise, then projects back down.

Where this fits in the modern Transformer block
-----------------------------------------------
    [ ... (Pre-Norm + Modern Attention + residual) ... ]
              |
    [ 3.1 RMSNorm 2                       ]
              |
    [ 3.3 SwiGLU FFN                      ]   <-- THIS FILE
              |
    [ + residual                          ]
              |
    [ Block output (B, T, d_model)        ]

GELU FFN (Part 1) vs SwiGLU (Part 3)
------------------------------------
    GELU FFN:   x -> Linear(d -> 4d) -> GELU -> Linear(4d -> d)
                two matmuls, one fixed nonlinearity, no gating.

    SwiGLU:     a = x @ W1                       # (B,T, 4d)   "value branch"
                b = SiLU(x @ W2)                  # (B,T, 4d)   "gate branch"
                y = (a * b) @ W3                  # (B,T,  d)   element-wise gated

Why gating?
-----------
The gate `b` is *content-dependent* — for each position and each hidden
unit, the network learns *whether* to let the value through. GELU gives a
fixed soft mask; SwiGLU lets the model decide per token. Empirically this
matches GELU's quality with fewer steps and scales better past 1B params.

SiLU (a.k.a. Swish):  silu(x) = x * sigmoid(x)
  Smooth, unbounded above, slight negative tail — same family as GELU
  but cheaper to compute.

Math
----
    SwiGLU(x) = ( (x W1)  *  silu(x W2) ) W3

W1 and W2 expand from d -> 4d (the standard "mult=4"); W3 contracts back.
No biases — the gate already gives positional flexibility.

Visualization
-------------
See notebook section 3.3:
  * SiLU vs GELU curve overlay (they're very similar)
  * activation map of `b` showing a few units gating different features
  * compare: same input through GELU FFN vs SwiGLU FFN

Shapes (with mult=4)
--------------------
  input  x   : (B, T, d_model)
  a, b each  : (B, T, 4 * d_model)
  a * b      : (B, T, 4 * d_model)
  output     : (B, T, d_model)

Parameter count (no biases, mult=4)
-----------------------------------
  W1, W2 : 4 * d_model^2  each   = 8 * d_model^2 total
  W3     : 4 * d_model^2
  total  : 12 * d_model^2         (vs 8 * d_model^2 for GELU FFN)
  --> SwiGLU has 50% more FFN params for similar quality. Many recipes
      shrink mult from 4 to 8/3 to keep total params equal to GELU.
"""
import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        a = self.w1(x)
        b = self.act(self.w2(x))
        return self.drop(self.w3(a * b))