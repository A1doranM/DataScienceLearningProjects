"""5.4 MoE layer — dispatch tokens to top-k experts, combine weighted outputs.

What this file does
-------------------
The full Mixture-of-Experts layer: a drop-in replacement for the dense
FFN sublayer of a Transformer block. Composes the router (5.1/5.2) with
E expert MLPs (5.3):

  1. flatten (B, T, C) -> (S, C) — routing is per *token*, batch ignored
  2. gate: idx (S, k), weights (S, k), aux loss
  3. dispatch: for each expert e and slot, select the tokens that chose
     e, run them through Expert_e
  4. combine: y[sel] += w[sel, slot] * Expert_e(x[sel])
  5. reshape back to (B, T, C); return (y, aux)

Where this fits in the Transformer block
----------------------------------------
The FFN slot — third treatment of the same sublayer across the course:

    [ Input tokens (B, T, C)              ]
              |
    [ RMSNorm 1                            ]
              |
    [ Modern Attention (Part 3)            ]
              |
    [ + residual                           ]
              |
    [ RMSNorm 2                            ]
              |
    [ FFN slot:                            ]
    [   Part 1: Linear-GELU-Linear (dense) ]
    [   Part 3: SwiGLU          (dense)    ]
    [   Part 5: MoE             (sparse)   ]   <-- THIS FILE
              |
    [ + residual                           ]
              |
    [ Block output (B, T, C) + aux loss    ]

Math (dispatch / combine, per token s)
--------------------------------------
    y_s = sum_{j=1..k}  w[s, j] * Expert_{idx[s, j]}( x_s )

Single-GPU implementation note: the double loop (experts x slots) with
boolean masks is O(E*k) Python iterations but each expert only processes
its own tokens — total work is still S*k expert passes. Production MoE
replaces the loop with batched all-to-all dispatch across GPUs
(expert parallelism, README 5.3). No capacity factor here: every routed
token is processed, none are dropped.

Training note
-------------
The aux loss must be added to the LM objective by the caller:
    total_loss = ce_loss + lambda_aux * aux        (lambda_aux ~ 0.01)

Visualization
-------------
See notebook section 5.4:
  * 4 anchor tokens traced through router -> experts -> combine
  * manual recomputation of one token's output verified against the layer
  * primary-expert load histogram (demo_moe.py prints the same)

Shapes (B=1, T=4, C=8, E=4, k=2)
--------------------------------
  input  x      : (1, 4, 8)
  x_flat        : (4, 8)        S = 4
  idx, w        : (4, 2) each
  per-expert x_e: (S_e, 8)      S_e = tokens routed to expert e
  y (combined)  : (4, 8) -> reshape (1, 4, 8)
  aux           : scalar

Parameter count
---------------
  router  : C*E + E              (negligible)
  experts : E * 12*C^2           (SwiGLU, mult=4)
  vs dense SwiGLU: 12*C^2  ->  E-fold capacity, k/E-fold relative compute
"""
from __future__ import annotations
import torch, torch.nn as nn
from gating import TopKGate
from experts import ExpertMLP

class MoE(nn.Module):
    """Mixture‑of‑Experts layer (token‑wise top‑k routing).
    Implementation is single‑GPU friendly (loops over experts for clarity).
    https://arxiv.org/pdf/2101.03961
    """
    def __init__(self, dim: int, n_expert: int, k: int = 1, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_expert = n_expert
        self.k = k
        self.gate = TopKGate(dim, n_expert, k=k)
        self.experts = nn.ModuleList([ExpertMLP(dim, mult=mult, swiglu=swiglu, dropout=dropout) for _ in range(n_expert)])

    def forward(self, x: torch.Tensor):
        """x: (B, T, C) → y: (B, T, C), aux_loss
        Steps: flatten tokens → gate → per‑expert forward → scatter back with weights.
        """
        B, T, C = x.shape
        S = B * T
        x_flat = x.reshape(S, C)
        idx, w, aux = self.gate(x_flat)  # (S,k), (S,k)

        y = torch.zeros_like(x_flat)     # (S,C)
        for e in range(self.n_expert):
            # tokens where expert e is selected at any of k slots
            for slot in range(self.k):
                sel = (idx[:, slot] == e)
                if sel.any():
                    x_e = x_flat[sel]
                    y_e = self.experts[e](x_e)
                    y[sel] += w[sel, slot:slot+1] * y_e
        y = y.view(B, T, C)
        return y, aux