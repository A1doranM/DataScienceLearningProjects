"""3.1 RMSNorm — Root Mean Square LayerNorm (Llama / Mistral style).

What this file does
-------------------
Implements RMSNorm: a simpler, faster cousin of LayerNorm that drops the
mean-subtraction and the learnable bias. It rescales each token vector so
its root-mean-square equals 1, then multiplies by a learnable per-dim gain.

Where this fits in the modern Transformer block
-----------------------------------------------
    [ Input tokens (B, T, d_model)        ]
              |
    [ 3.1 RMSNorm 1                       ]   <-- THIS FILE
              |
    [ 3.5 Modern Attention                ]
        (RoPE inside / GQA / sliding-win  )
              |
    [ + residual                          ]
              |
    [ 3.1 RMSNorm 2                       ]   <-- THIS FILE (second use)
              |
    [ 3.3 SwiGLU FFN                      ]
              |
    [ + residual                          ]
              |
    [ Block output (B, T, d_model)        ]

LayerNorm vs RMSNorm
--------------------
    LayerNorm:    y = gamma * (x - mu) / sqrt(var + eps) + beta
                  ^ subtracts mean,  ^ scales by std,    ^ learned bias

    RMSNorm:      y = g     *  x      /  rms(x)
                  ^ no bias  ^ no mean subtraction
                                 rms(x) = sqrt( mean(x^2) + eps )

Why drop the mean? Empirically the mean centering does little once the
network learns to handle absolute magnitude — what matters is bringing the
*scale* of each token vector under control. Removing it makes RMSNorm
~7-15% faster per call and slightly fewer parameters (no beta).

Math
----
    rms(x) = sqrt( (1/d) * sum_i x_i^2 + eps )
    y      = (x / rms(x)) * g            # g is shape (d,) learned

Visualization
-------------
See notebook section 3.1:
  * heatmap of one (T, d_model) tensor before vs after RMSNorm
  * compare with LayerNorm on the same input

Shapes
------
  input  x : (B, T, d_model)       — same as LayerNorm
  output y : (B, T, d_model)       — shape preserved
  weight g : (d_model,)            — only learnable parameter

Parameter count
---------------
  RMSNorm:   d_model            (gain only)
  LayerNorm: 2 * d_model         (gain + bias)
"""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    y = x * g / rms(x),   rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight