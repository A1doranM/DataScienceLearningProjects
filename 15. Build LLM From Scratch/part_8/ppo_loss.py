"""8.5 PPO Loss — the clipped surrogate objective that actually trains the policy.

What this file does
-------------------
This is the heart of PPO. After we have rolled out completions, scored them with
the reward model, applied the KL leash to the reference, and turned everything into
a per-token `advantage`, this file converts that signal into three scalar losses
and bundles them into one number we can call `.backward()` on.

The trick is the *clipped surrogate*. We compare the NEW policy (the one we are
updating right now) to the OLD policy (the snapshot that generated the data) via a
probability ratio. If the new policy drifts too far in one update, we clip the
ratio so the objective stops rewarding further drift — a soft "trust region" that
keeps each update small and stable. In pseudo-code::

    ratio       = exp(new_logp - old_logp)          # how much more/less likely now
    unclipped   = ratio * adv
    clipped     = clamp(ratio, 1-clip, 1+clip) * adv
    policy_loss = -mean( min(unclipped, clipped) )  # minus: we maximize, optimizer minimizes
    value_loss  = MSE(new_values, returns)          # critic learns to predict returns
    total       = policy_loss + vf_coef*value_loss - ent_coef*entropy

Why `min`? The clip is *one-sided per the sign of the advantage*. For a good action
(adv > 0) we cap how much the loss can reward pushing the ratio above 1+clip; for a
bad action (adv < 0) we cap how much it rewards pushing the ratio below 1-clip.
Taking the elementwise `min` of the unclipped and clipped terms is the algebra that
flattens the objective past the clip band in whichever direction the advantage
points — so once a token's probability has moved "enough" this step, the gradient
through it goes to zero and the update can't blow up.

Where this fits in the Part 8 RLHF-PPO loop
-------------------------------------------
    prompt
       |
    [ policy does a take: generate + record old_logp   (policy.py / rollout.py) ]
       |
    [ judge scores the take: reward model r            (part_7 RewardModel)     ]
       |
    [ KL leash: shaped_r = r - kl * (logp - ref_logp)  (vs frozen SFT reference)]
       |
    [ advantage = shaped_r - value baseline, normalized (train_ppo.py)          ]
       |
    [ PPO clipped update: min(unclipped, clipped)      (ppo_loss.py)            ]   <-- THIS FILE
       |
    [ AdamW step on the policy only                    (train_ppo.py)           ]
       |
    [ eval: avg reward, tuned policy vs frozen ref     (eval_ppo.py)            ]

Math
----
  Let i index the N flattened (sample, token) positions in the batch.

  ratio_i        = exp(new_logp_i - old_logp_i)
      new_logp_i = log prob the CURRENT policy assigns the taken token
      old_logp_i = log prob the SNAPSHOT policy gave it (constant, no grad)
      adv_i      = advantage for that token (good = positive, bad = negative)

  unclipped_i    = ratio_i * adv_i
  clipped_i      = clamp(ratio_i, 1 - clip, 1 + clip) * adv_i
      clip       = clip_ratio (default 0.2 → ratio pinned to [0.8, 1.2])

  policy_loss    = - (1/N) * sum_i min(unclipped_i, clipped_i)
      the leading minus: PPO maximizes the surrogate, optimizers minimize loss

  value_loss     = (1/N) * sum_i (new_values_i - returns_i)^2          (plain MSE)
      new_values = critic's value estimate now;  returns = bootstrapped targets

  entropy        = - (1/N) * sum_i new_logp_i      (cheap APPROX entropy bonus;
                   true entropy needs the full vocab distribution, not one logp)

  approx_kl      = (1/N) * sum_i (old_logp_i - new_logp_i)   (logging only, no grad use)

  total          = policy_loss + vf_coef * value_loss - ent_coef * entropy
      vf_coef    = 0.5 (weight on critic loss),  ent_coef = 0.0 (entropy off by default)

Visualization
-------------
See notebook section 8.5 — plots the clipped surrogate against the ratio for a
positive vs negative advantage, showing where the objective flattens at 1-clip and
1+clip and why the gradient vanishes outside the trust band.

Shapes
------
  new_logp, old_logp  (N,)   per-token log-probs; N = total tokens across the batch
  adv                 (N,)   per-token advantages (already normalized upstream)
  new_values          (N,)   critic value predictions for those positions
  old_values          (N,)   snapshot value predictions (accepted but unused here)
  returns             (N,)   value targets the critic regresses toward
  ratio               (N,)   exp(new_logp - old_logp)
  unclipped, clipped  (N,)   ratio * adv  and  clamped-ratio * adv
  policy_loss         ()     scalar, mean over N
  value_loss          ()     scalar, MSE over N
  entropy             ()     scalar, -mean(new_logp)
  approx_kl           ()     scalar, mean(old_logp - new_logp)
  total_loss          ()     scalar fed to .backward()
"""

from __future__ import annotations
import torch, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class PPOLossOut:
    # Bundle of the four scalar losses + the KL diagnostic, returned together so
    # train_ppo.py can both backprop `total_loss` and log the individual pieces.
    policy_loss: torch.Tensor   # () clipped-surrogate loss (drives the actor)
    value_loss: torch.Tensor    # () critic MSE against returns
    entropy: torch.Tensor       # () approx entropy bonus term (off by default)
    approx_kl: torch.Tensor     # () mean(old_logp - new_logp), logging only
    total_loss: torch.Tensor    # () policy + vf_coef*value - ent_coef*entropy


def ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
               clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0):
    # policy
    ratio = torch.exp(new_logp - old_logp)  # (N,) new/old prob per token; ==1 before any update
    unclipped = ratio * adv                 # (N,) raw surrogate: reward moving good actions up
    # (N,) clamp ratio to [1-clip, 1+clip] (default [0.8, 1.2]) so one update can't drift too far
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    # min() picks whichever branch is the pessimistic (lower) reward → flattens objective past the
    # clip band on the side the advantage points; mean over N, minus sign because we maximize.
    policy_loss = -torch.mean(torch.min(unclipped, clipped))  # () scalar

    # value (clip optional → here: simple MSE)
    value_loss = F.mse_loss(new_values, returns)  # () critic regresses predictions toward returns

    # entropy bonus (we approximate entropy via -new_logp mean; strictly needs full dist)
    entropy = -new_logp.mean()  # () cheap proxy; with ent_coef=0 default it does not affect total

    # approx KL for logging
    approx_kl = torch.mean(old_logp - new_logp)  # () drift diagnostic only (not part of the loss)

    # () weighted sum: actor + vf_coef*critic - ent_coef*entropy (defaults 0.5 / 0.0)
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return PPOLossOut(policy_loss, value_loss, entropy, approx_kl, total)