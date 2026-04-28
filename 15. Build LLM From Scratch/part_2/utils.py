"""2.5 Sampling utilities -- temperature support via top-k and top-p filters.

What this file does
-------------------
Filters next-token logits before sampling. The model produces one score per
byte token; these helpers optionally keep only the top-k tokens or a nucleus-
style high-probability prefix controlled by top_p.

Where this fits in the Part 2 training pipeline
-----------------------------------------------
    [ GPT context ]
          |
    [ logits for next token (B, 256) ]
          |
    [ temperature scaling ]
          |
    [ top-k / top-p filtering ]       <-- THIS FILE
          |
    [ softmax + multinomial sample ]

Connection to Part 1 and later parts
------------------------------------
Part 1 explained the forward computation inside a block. This file sits after
the block stack: it decides how adventurous generation should be. Later parts
reuse the same idea when sampling SFT, PPO, or GRPO policies.

Math
----
Top-k:
    keep token j if logit_j is among the k largest logits

Top-p / nucleus-style filtering in this file:
    sort tokens by probability p_j
    mask tokens after the cumulative probability passes top_p
    always keep the highest-probability token

Filtered tokens receive -infinity so softmax gives them probability 0.

Shapes
------
    logits:          FloatTensor [B, vocab_size]
    filtered logits: FloatTensor [B, vocab_size]

Visualization
-------------
See notebook section 2.5 -- Sampling. It shows a toy probability row before
and after temperature, top-k, and top-p filtering.
"""

from __future__ import annotations
import torch

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    - logits: (B, vocab)
    Returns filtered logits with -inf for masked entries.
    """
    B, V = logits.shape
    filtered = logits.clone()

    if top_k is not None and top_k < V:
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)
        kth = topk_vals[:, -1].unsqueeze(-1)
        # Anything below the kth-largest logit becomes impossible after softmax.
        filtered[filtered < kth] = float('-inf')

    if top_p is not None and 0 < top_p < 1.0:
        # Sort high-to-low so cumulative probability forms the nucleus prefix.
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum > top_p
        # keep at least 1 token
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')
        # Scatter back
        filtered = torch.full_like(filtered, float('-inf'))
        filtered.scatter_(1, sorted_idx, sorted_logits)

    return filtered
