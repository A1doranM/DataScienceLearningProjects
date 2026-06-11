"""5.5 HybridFFN — blend a dense FFN with an MoE: y = α·Dense(x) + (1−α)·MoE(x).

What this file does
-------------------
A middle ground between a fully dense FFN and a fully sparse MoE. Both
paths process every token; their outputs are mixed with a fixed scalar
alpha in [0, 1]:

    alpha = 1.0  -> pure dense   (stable, lower capacity)
    alpha = 0.0  -> pure MoE     (max capacity, routing noise)
    alpha = 0.5  -> equal blend

Where this fits in the Transformer block
----------------------------------------
Drop-in for the same FFN slot as moe.py:

    [ RMSNorm 2                          ]
              |
    [ 5.5 HybridFFN                      ]   <-- THIS FILE
    [   +-- Dense GELU MLP   (always on) ]
    [   +-- 5.4 MoE          (routed)    ]
    [   y = α·dense + (1−α)·moe          ]
              |
    [ + residual                          ]

Why hybrid?
-----------
1. Surface: dense path guarantees every token gets *some* useful FFN
   even when the router misroutes early in training.
2. Practical: real architectures rarely make every layer MoE — they
   alternate dense/MoE layers (Switch, Mixtral use other patterns) or
   keep a shared expert that is always active (DeepSeek-MoE). The blend
   here is the simplest form of "shared + routed" capacity.
3. Deep: the dense path provides a low-variance gradient signal that
   stabilizes the noisy, discrete routing decisions of the MoE path.

Math
----
    y = alpha * Dense(x) + (1 - alpha) * MoE(x)
    Dense = Linear(C -> 4C) -> GELU -> Linear(4C -> C)
    aux is passed through from the MoE unchanged.

Visualization
-------------
See notebook section 5.5 — same input through alpha = 0 / 0.5 / 1.

Shapes
------
  input  x : (B, T, C)
  output y : (B, T, C)   + aux scalar (from the MoE path)
"""
from __future__ import annotations
import torch.nn as nn
from moe import MoE

class HybridFFN(nn.Module):
    """Blend dense FFN with MoE output: y = α * Dense(x) + (1−α) * MoE(x).
    Use α∈[0,1] to trade between stability (dense) and capacity (MoE).
    """
    def __init__(self, dim: int, alpha: float = 0.5, mult: int = 4, swiglu: bool = True, n_expert: int = 4, k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.alpha = alpha
        inner = mult * dim
        self.dense = nn.Sequential(
            nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, dim), nn.Dropout(dropout)
        )
        self.moe = MoE(dim, n_expert=n_expert, k=k, mult=mult, swiglu=swiglu, dropout=dropout)
    def forward(self, x):
        y_dense = self.dense(x)
        y_moe, aux = self.moe(x)
        y = self.alpha * y_dense + (1.0 - self.alpha) * y_moe
        return y, aux