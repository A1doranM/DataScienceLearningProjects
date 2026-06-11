"""5.3 ExpertMLP — one expert: a SwiGLU (or GELU) feed-forward network.

What this file does
-------------------
Defines a single expert. Mathematically it is *exactly* Part 3's SwiGLU
FFN (or Part 1's GELU MLP with swiglu=False) — only the attribute names
differ (inp1/inp2/out here vs w1/w2/w3 in part_3/swiglu.py):

    a = x @ inp1                      # value branch     (S_e, mult*C)
    b = SiLU(x @ inp2)                # gate branch      (S_e, mult*C)
    y = (a * b) @ out                 # combine + project (S_e, C)

An MoE layer holds E of these in an nn.ModuleList; each token runs
through only the k experts its router picked.

Where this fits in the MoE layer
--------------------------------
    [ x (B, T, C) ] -> flatten -> (S, C)
              |
    [ 5.1 TopKGate router  -> idx, w, aux ]
              |
    [ 5.4 dispatch:                       ]
    [   Expert_0   Expert_1 ... Expert_E-1 ]   <-- THIS FILE (xE copies)
    [   each processes only its tokens     ]
              |
    [ weighted combine -> y (B, T, C)     ]

The capacity-vs-compute trick
-----------------------------
E experts hold E * 12*C^2 parameters (SwiGLU, mult=4), but each token
only *executes* k of them. So:

    parameters (capacity) grow with E      — what the model can store
    FLOPs/token (compute) grow with k      — what you pay per token

That decoupling is the whole point of MoE: a 8-expert k=2 layer has 8x
the FFN capacity at 2x the dense FFN compute.

Visualization
-------------
See notebook section 5.3 — params-vs-FLOPs bar chart for dense vs MoE.

Shapes (one expert, mult=4)
---------------------------
  input  x  : (S_e, C)      S_e = tokens routed to this expert
  a, b      : (S_e, 4*C)
  output    : (S_e, C)

Parameter count (per expert, SwiGLU, no biases)
-----------------------------------------------
  inp1, inp2 : 4*C^2 each
  out        : 4*C^2
  total      : 12*C^2   ->  MoE layer total: E * 12*C^2 (+ tiny router)
"""
from __future__ import annotations
import torch.nn as nn

class ExpertMLP(nn.Module):
    """Single expert MLP (SwiGLU or GELU)."""
    def __init__(self, dim: int, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        if swiglu:
            self.inp1 = nn.Linear(dim, inner, bias=False)
            self.inp2 = nn.Linear(dim, inner, bias=False)
            self.act = nn.SiLU()
            self.out = nn.Linear(inner, dim, bias=False)
            self.drop = nn.Dropout(dropout)
            self.swiglu = True
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, dim), nn.Dropout(dropout)
            )
            self.swiglu = False
    def forward(self, x):
        if self.swiglu:
            a = self.inp1(x); b = self.act(self.inp2(x))
            return self.drop(self.out(a * b))
        else:
            return self.ff(x)