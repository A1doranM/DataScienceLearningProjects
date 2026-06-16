"""7.3 Preference loss — turn two scalar rewards into one number to minimize.

What this file does
-------------------
This is the heart of Part 7. The reward model (model_reward.py) has already
turned the *chosen* answer into a scalar r_pos and the *rejected* answer into a
scalar r_neg. All a preference loss has to do is push r_pos above r_neg. It
never looks at the raw scores on their own — only at the gap between them:

    gap = r_pos - r_neg          # how much we prefer chosen over rejected

The Bradley-Terry loss (the main one) is simply the negative log-probability
that the chosen answer wins, under a logistic (sigmoid) model of "who wins":

    loss = softplus(-gap) = -log sigmoid(gap)

Because it depends only on the gap, adding the same constant to *both* rewards
changes nothing — exactly like Elo ratings in chess, where only rating
*differences* decide win probabilities. The optional margin_ranking_loss is a
simpler hinge alternative that just wants the gap to exceed a fixed margin.

Where this fits in the Part 7 reward-modeling pipeline
------------------------------------------------------
    preference pair (prompt, chosen, rejected)        (data_prefs.py)
              |
    [ plate both with SFT template + tokenize         (collator_rm.py)  ]
    [   -> (pos_ids, neg_ids)                                           ]
              |
    [ reward model: encoder -> mean-pool -> scalar    (model_reward.py) ]
    [   r_pos = score(chosen),  r_neg = score(rejected)                 ]
              |
    [ Bradley-Terry loss on the gap                   (loss_reward.py)  ]   <-- THIS FILE
    [   softplus(-(r_pos - r_neg))                                      ]
              |
    [ train -> reward checkpoint                      (train_rm.py)     ]
              |
    [ eval: pairwise accuracy r_pos > r_neg           (eval_rm.py)      ]

Math
----
Let gap = r_pos - r_neg.

Bradley-Terry (a.k.a. logistic / DPO-style reward loss):
    bradley_terry_loss = softplus(-gap) = -log sigmoid(gap)
      softplus(z) = log(1 + e^z)        (smooth, always >= 0)
      sigmoid(gap) = P(chosen beats rejected) under the Bradley-Terry model
    Behavior of the loss as the gap changes:
      gap -> +inf  (chosen clearly better)  : loss -> 0       (nothing to fix)
      gap  =  0    (a tie)                  : loss = log 2 ~= 0.693
      gap -> -inf  (ranking is backwards)   : loss ~= -gap    (grows linearly)
    Depends ONLY on the gap: shifting r_pos and r_neg by the same amount leaves
    the loss unchanged (Elo-like — only the difference matters).

Margin ranking (hinge alternative):
    margin_ranking_loss = max(0, margin - gap)
      zero once gap >= margin (chosen wins by at least `margin`);
      otherwise penalizes linearly. Sharp cutoff instead of a smooth curve.

The `.mean()` / built-in reduction averages the per-pair losses over the batch.

Visualization
-------------
See notebook section 7.3 — plots the loss vs. the gap (r_pos - r_neg), showing
the smooth Bradley-Terry curve passing through log 2 at a tie versus the hinge.

Shapes
------
  r_pos : (B,)      one scalar reward per chosen answer in the batch
  r_neg : (B,)      one scalar reward per rejected answer in the batch
  gap   : (B,)      elementwise r_pos - r_neg
  loss  : ()        scalar after reduction over the batch
"""
from __future__ import annotations
import torch, torch.nn.functional as F

def bradley_terry_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    """-log sigma(r_pos - r_neg) = softplus(-(r_pos - r_neg))
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html"""
    diff = r_pos - r_neg                # (B,) gap; > 0 means we prefer chosen
    return F.softplus(-diff).mean()     # softplus(-gap) = -log sigmoid(gap), averaged over batch -> ()


def margin_ranking_loss(r_pos: torch.Tensor, r_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html"""
    y = torch.ones_like(r_pos)          # (B,) all +1: target says "r_pos should rank above r_neg"
    return F.margin_ranking_loss(r_pos, r_neg, y, margin=margin)  # max(0, margin - (r_pos - r_neg)), mean -> ()