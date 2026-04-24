"""Walkthrough: prints every MHA intermediate tensor's shape, step by step.

What this file does
-------------------
Runs one forward pass through MultiHeadSelfAttention with B=1, T=5, d_model=12,
n_head=3, and prints the shape of every intermediate tensor — before and after
the reshape / transpose / matmul / softmax / merge steps. Also writes the log
to part_1/out/mha_shapes.txt so you can re-read it later.

Use this as a checklist when reading multi_head.py — trace each print line
back to the source to verify you understand the reshapes.

Where this fits in the Transformer block
----------------------------------------
Exercises the MHA block specifically:

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

What you'll see printed
-----------------------
    Input x:           (1, 5, 12)     = (B, T, d_model)
    Linear qkv(x):     (1, 5, 36)     = (B, T, 3*d_model)
    view to 5D:        (1, 5, 3, 3, 4)= (B, T, 3, heads, d_head)
    q,k,v split:       each (1, 5, 3, 4)
    transpose heads:   each (1, 3, 5, 4) = (B, heads, T, d_head)
    scores q@k^T:      (1, 3, 5, 5)   = (B, heads, T, T)
    softmax(weights):  (1, 3, 5, 5)
    context @v:        (1, 3, 5, 4)   = (B, heads, T, d_head)
    merge heads:       (1, 5, 12)     = (B, T, d_model)
    final proj:        (1, 5, 12)     = (B, T, d_model)

How to run
----------
    cd part_1
    python demo_mha_shapes.py
"""

import os
import math
import torch
from multi_head import MultiHeadSelfAttention

OUT_TXT = os.path.join(os.path.dirname(__file__), 'out', 'mha_shapes.txt')


def log(s):
    print(s)
    with open(OUT_TXT, 'a') as f:
        f.write(s + "\n")


if __name__ == "__main__":
    # Reset file
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, 'w').close()

    B, T, d_model, n_head = 1, 5, 12, 3
    d_head = d_model // n_head
    x = torch.randn(B, T, d_model)
    attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=True)

    log(f"Input x:           {tuple(x.shape)} = (B,T,d_model)")
    qkv = attn.qkv(x)  # (B,T,3*d_model)
    log(f"Linear qkv(x):     {tuple(qkv.shape)} = (B,T,3*d_model)")

    qkv = qkv.view(B, T, 3, n_head, d_head)
    log(f"view to 5D:        {tuple(qkv.shape)} = (B,T,3,heads,d_head)")

    q, k, v = qkv.unbind(dim=2)
    log(f"q,k,v split:       q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    log(f"transpose heads:   q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} = (B,heads,T,d_head)")

    scale = 1.0 / math.sqrt(d_head)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    log(f"scores q@k^T:      {tuple(scores.shape)} = (B,heads,T,T)")

    weights = torch.softmax(scores, dim=-1)
    log(f"softmax(weights):  {tuple(weights.shape)} = (B,heads,T,T)")

    ctx = torch.matmul(weights, v)
    log(f"context @v:        {tuple(ctx.shape)} = (B,heads,T,d_head)")

    out = ctx.transpose(1, 2).contiguous().view(B, T, d_model)
    log(f"merge heads:       {tuple(out.shape)} = (B,T,d_model)")

    out = attn.proj(out)
    log(f"final proj:        {tuple(out.shape)} = (B,T,d_model)")

    log("\nLegend:")
    log("  B=batch, T=sequence length, d_model=embedding size, heads=n_head, d_head=d_model/heads")
    log("  qkv(x) is a single Linear producing [Q|K|V]; we reshape then split into q,k,v")