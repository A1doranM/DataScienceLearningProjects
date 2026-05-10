"""3.2 RoPE — Rotary Position Embedding (precompute + apply).

What this file does
-------------------
Implements Rotary Position Embeddings: position is injected by *rotating*
each (x_2i, x_2i+1) feature pair of Q and K by a position-dependent angle.
There is no PE table to add — instead, the dot product Q·K^T naturally
becomes a function of the *relative* offset (j - i).

  RoPECache       — precomputes cos/sin tables for positions 0..max_pos-1
  apply_rope_single — rotates a single tensor (Q or K) given cos/sin

Where this fits in the modern Transformer block
-----------------------------------------------
    [ Input tokens (B, T, d_model)        ]
              |
    [ 3.1 RMSNorm 1                       ]
              |
    [ 3.5 Modern Attention                ]
        - project to Q, K, V
        - apply RoPE to Q, K        <-- THIS FILE
        - scaled-dot-product attention
              |
    [ + residual                          ]
              |
    [ ... (rest of block) ...             ]

Why rotate in pairs?
--------------------
A 2D rotation by angle theta is the linear map:
    (x, y) -> (x*cos - y*sin,  x*sin + y*cos)

Applied at position p with theta_i = p * inv_freq[i], this gives each
feature pair its own "clock hand" that ticks at a different rate as p
changes. Then for two positions p and q:

    < RoPE(q_p), RoPE(k_q) >  =  cos((q-p)*theta_i) * (q . k)  + ...

i.e. the score depends only on (q - p), the *relative* offset — exactly
what attention should care about.

Compared to additive sinusoidal PE (Part 1), RoPE:
  * works at any sequence length without retraining (extrapolation)
  * lets you grow the cache lazily (`_build` doubles when needed)
  * does not need a separate add-step in the model forward

Math
----
For head_dim D (must be even), define inv_freq for i = 0, 2, 4, ..., D-2:
    inv_freq[i] = 1 / 10000 ^ ( i / D )

For position p:
    theta_i_p   = p * inv_freq[i]
    cos[p, i/2] = cos(theta_i_p)
    sin[p, i/2] = sin(theta_i_p)

Applied to a vector x of length D, split into pairs (x_0,x_1), (x_2,x_3) ...:
    x_{2i}'   = x_{2i}  * cos - x_{2i+1} * sin
    x_{2i+1}' = x_{2i}  * sin + x_{2i+1} * cos

Visualization
-------------
See notebook section 3.2:
  * cos/sin tables as heatmaps over (max_pos, D/2)
  * one query vector rotated through positions 0..7 (vector field)
  * dot product < RoPE(q_p), RoPE(k_q) > vs (q - p) — only depends on offset

Shapes
------
  inv_freq        : (D/2,)
  cos / sin tables: (max_pos, D/2)
  cos / sin slice : (T, D/2)            for positions [start..start+T-1]
  input  q or k   : (B, H, T, D)        D even
  output rotated  : (B, H, T, D)        same shape

Parameter count
---------------
  Zero! The cos/sin tables are deterministic functions of position.
"""
from __future__ import annotations
import torch
import math

class RoPECache:
    """Precompute cos/sin for positions up to max_pos for even head_dim."""
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self._build(max_pos)
    def get(self, positions: torch.Tensor):
        # positions: (T,) or (1,T)
        if positions.dim() == 2:
            positions = positions[0]
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            # grow tables
            self._build(max(need, int(self.max_pos * 2)))
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        return cos, sin
    
    def _build(self, max_pos: int):
        """(Re)build cos/sin tables for a new max_pos."""
        self.max_pos = max_pos
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        t = torch.arange(max_pos, device=self.device).float()
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs along last dim for RoPE.
    x: (B,H,T,D) with D even; cos/sin: (T,D/2)
    """
    assert x.size(-1) % 2 == 0
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out
