"""Visualize multi-head attention weights per head (grid).

What this file does
-------------------
Runs a forward pass through MultiHeadSelfAttention (B=1, T=5, d_model=12,
n_head=3) on a random input, then saves the per-head (T, T) attention
weight matrices as a single grid figure to  part_1/out/multi_head_attn_grid.png.

Each subplot in the grid corresponds to one attention head and shows the
(query_pos, key_pos) attention map — darker = more attention. Even with
random initialization, different heads produce different patterns; after
training on real text, these patterns become meaningful (local, syntactic,
punctuation-attending, etc.).

Where this fits in the Transformer block
----------------------------------------
  [ Input tokens (B, T, d_model) ]
                 |
  [ 1.1 Positional Encoding      ]
                 |
  [ 1.5 LayerNorm 1              ]
                 |
==> 1.3/1.4 Multi-Head Attention ]
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

What the output image shows
---------------------------
  * rows  = query positions (who's looking)
  * cols  = key positions (what's being looked at)
  * Because the MHA applies a causal mask, the attention weights are
    lower-triangular: upper-right of each heatmap is effectively zero.

How to run
----------
    cd part_1
    python demo_visualize_multi_head.py
"""

import torch
from multi_head import MultiHeadSelfAttention
from vis_utils import save_attention_heads_grid

B, T, d_model, n_head = 1, 5, 12, 3
x = torch.randn(B, T, d_model)
attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=False)

out, w = attn(x)  # w: (B, H, T, T)

save_attention_heads_grid(w.detach().cpu().numpy(), filename="multi_head_attn_grid.png")