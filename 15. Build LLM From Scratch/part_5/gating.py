"""5.1/5.2 TopKGate — the router: top-k softmax gating + load-balancing aux loss.

What this file does
-------------------
Implements the *router* of a Mixture-of-Experts layer. For every token it:

  1. scores all E experts with a single Linear(dim -> E)        (5.1)
  2. softmaxes the scores into a probability distribution
  3. picks the top-k experts (k = 1 or 2 typical) and returns
     their ids + gate weights
  4. computes the Switch-Transformer load-balancing aux loss     (5.2)
     that keeps experts evenly used

Where this fits in the MoE layer
--------------------------------
    [ x (B, T, C)  — attention output, after RMSNorm ]
              |
    [ flatten -> (S, C),  S = B*T tokens             ]
              |
    [ 5.1 TopKGate router                            ]   <-- THIS FILE
        |          |
        |          +--> 5.2 aux loss (scalar)            <-- THIS FILE
        v
    [ idx (S, k)  +  weights (S, k)                  ]
              |
    [ 5.4 dispatch -> experts -> weighted combine    ]
              |
    [ y (B, T, C)                                    ]

Math (5.1 — routing)
--------------------
    logits = x @ W_g + b              # (S, E)
    p      = softmax(logits, dim=-1)  # (S, E)   each row sums to 1
    w, idx = topk(p, k)               # (S, k)   raw probs, NOT renormalized

NOTE: weights are the raw top-k softmax probabilities, so per token they
sum to <= 1 (with k=1 the expert output is scaled by the router's
confidence). Many implementations renormalize w to sum to 1; this one
deliberately keeps the Switch-style raw probabilities.

Math (5.2 — load-balancing aux loss, Switch Transformer eq. 4-6)
----------------------------------------------------------------
    importance_e = mean_over_tokens( p[:, e] )      # soft usage  (differentiable)
    load_e       = #{tokens whose top-1 == e} / S   # hard usage  (not differentiable)
    L_aux        = E * sum_e( importance_e * load_e )

  * perfectly balanced  -> importance_e = load_e = 1/E  -> L_aux = 1.0
  * fully collapsed     -> one expert has importance ~= load ~= 1 -> L_aux ~= E

Gradients flow through `importance` (softmax probs); `load` acts as a
per-expert scaling. Add `lambda * L_aux` (lambda ~ 0.01) to the LM loss.

Visualization
-------------
See notebook section 5.1/5.2:
  * (S, E) routing-probability heatmap with chosen experts outlined
  * balanced vs collapsed router bar charts with hand-computed aux values

Shapes
------
  input  x   : (S, C)        S = B*T flattened tokens
  logits     : (S, E)
  probs      : (S, E)
  idx        : (S, k)  long  — expert ids, slot 0 is the primary expert
  weights    : (S, k)  float — raw softmax probs (sum <= 1 per token)
  aux_loss   : scalar

Parameter count
---------------
  W_g : dim * n_expert + n_expert   — tiny (the router is ~free)
"""
from __future__ import annotations
import torch, torch.nn as nn

class TopKGate(nn.Module):
    """Top‑k softmax gating with Switch‑style load‑balancing aux loss.
    Args:
      dim: input hidden size
      n_expert: number of experts
      k: number of experts to route per token (1 or 2 typical)
    Returns:
      (indices, weights, aux_loss) where
        indices: (S, k) long, expert ids for each token
        weights: (S, k) float, gate weights (sum ≤ 1 per token)
        aux_loss: scalar load‑balancing penalty
    """
    def __init__(self, dim: int, n_expert: int, k: int = 1):
        super().__init__()
        assert k >= 1 and k <= n_expert
        self.n_expert = n_expert
        self.k = k
        self.w_g = nn.Linear(dim, n_expert, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (S, C) where S = tokens (batch * seq)
        logits = self.w_g(x)                  # (S, E)
        probs = torch.softmax(logits, dim=-1) # (S, E)
        topk_vals, topk_idx = torch.topk(probs, k=self.k, dim=-1)  # (S,k)

        # Load‑balancing aux loss (Switch):
        S, E = probs.size(0), probs.size(1)
        # importance: avg prob per expert
        importance = probs.mean(dim=0)                 # (E,)
        # load: fraction of tokens assigned as primary (top‑1 hard assignment)
        hard1 = topk_idx[:, 0]                         # (S,)
        load = torch.zeros(E, device=x.device)
        load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype))
        load = load / max(S, 1)
        aux_loss = (E * (importance * load).sum())
        # print("*"*50)
        # print(probs, importance, hard1, load, aux_loss)
        # print("*"*50)

        return topk_idx, topk_vals, aux_loss