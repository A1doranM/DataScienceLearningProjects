"""9.5 GRPO Loss — PPO's clipped surrogate with the critic removed and the KL moved into the loss.

What this file does
-------------------
This is Part 8's `ppo_loss.py` with two things taken away and one thing added back
in a new place. After train_grpo.py has rolled out a GROUP of completions, scored
each, and turned them into a per-token `advantage` (advantage = reward minus the
group's average reward, see notebook 9.4), this file converts that signal into the
scalar loss we call `.backward()` on.

What changed from PPO (ppo_loss.py):
  - The clipped surrogate is IDENTICAL (ratio, clamp, min -- the trust region).
  - The value loss is GONE: there is no critic to regress, so no MSE(new_values, returns).
  - The KL is now a SEPARATE PENALTY TERM in the loss: total = L_PPO + kl_coef * KL(pi||ref).
    (In PPO the KL shaped the *reward* upstream; here it is added to the *loss*.)
  - The entropy bonus is present but off by default (ent_coef=0.0 -- "not used in the
    original GRPO paper", per the inline note).

In pseudo-code::

    ratio       = exp(new_logp - old_logp)              # same as PPO
    unclipped   = ratio * adv
    clipped     = clamp(ratio, 1-clip, 1+clip) * adv
    policy_loss = -mean( min(unclipped, clipped) )      # same clipped surrogate as PPO
    total       = policy_loss - ent_coef*entropy + kl_coef*kl_ref   # KL is a loss term now

Why `min`? Identical reasoning to PPO: the clip is one-sided per the sign of the
advantage, so taking the elementwise `min` flattens the objective past the clip band
in whichever direction the advantage points -- once a token has moved "enough" this
step, its gradient goes to zero and the update can't blow up.

Where this fits in the Part 9 RLHF-GRPO loop
--------------------------------------------
    prompt
       |
    [ policy does a GROUP of takes: G generations + old_logp  (policy.py / rollout.py) ]
       |
    [ judge scores each take: reward r_1..r_G                 (part_7 RewardModel)     ]
       |
    [ group baseline: A_i = r_i - mean(r over the group)      (train_grpo.py)          ]
       |
    [ clipped surrogate: min(unclipped, clipped)             (grpo_loss.py)           ]   <-- THIS FILE
       |
    [ + KL penalty in the loss: kl_coef * KL(new || ref)     (grpo_loss.py)           ]   <-- THIS FILE
       |
    [ AdamW step on the policy only (no value head)          (train_grpo.py)          ]
       |
    [ eval: avg reward, tuned policy vs frozen ref           (eval_ppo.py)            ]

Math
----
  Let i index the N flattened action tokens in the batch.

  ratio_i        = exp(new_logp_i - old_logp_i)
      new_logp_i = log prob the CURRENT policy assigns the taken token (has grad)
      old_logp_i = log prob the SNAPSHOT policy gave it (constant, no grad)
      adv_i      = group-relative advantage for that token, broadcast from its take

  unclipped_i    = ratio_i * adv_i
  clipped_i      = clamp(ratio_i, 1 - clip, 1 + clip) * adv_i      (clip = clip_ratio, default 0.2)
  policy_loss    = - (1/N) * sum_i min(unclipped_i, clipped_i)     (minus: we maximize)

  entropy        = - (1/N) * sum_i new_logp_i   if ent_coef != 0 else 0   (off by default)
  approx_kl      = (1/N) * sum_i (old_logp_i - new_logp_i)   (logging only; how far we moved)
  kl_ref         = kl_mean   (passed in: the mean KL(pi_new || pi_ref) on action tokens)

  total          = policy_loss - ent_coef * entropy + kl_coef * kl_ref
      NO value loss term -- the critic is gone.  kl_coef defaults to 0.0 here but
      train_grpo.py passes kl_coef = 0.01 with kl_mean = mean(new_logp - ref_logp).

  Two different KLs, do not confuse them:
      approx_kl = mean(old_logp - new_logp)   -> DIAGNOSTIC (drift from the snapshot), not in the loss
      kl_ref    = mean(new_logp - ref_logp)   -> PENALTY (drift from the frozen SFT ref), in the loss

Visualization
-------------
See notebook section 9.5 — plots the clipped surrogate vs the ratio for a positive
(above group average) and negative (below) advantage, showing it flattens at 1-clip
and 1+clip exactly as in Part 8; only the baseline that produces `adv` changed.

Shapes
------
  new_logp, old_logp  (N,)   per-token log-probs on action tokens; N = total action tokens
  adv                 (N,)   per-token group-relative advantages (already normalized upstream)
  kl_mean             ()     scalar mean KL(pi||ref) over action tokens (or None -> 0)
  ratio               (N,)   exp(new_logp - old_logp)
  unclipped, clipped  (N,)   ratio * adv  and  clamped-ratio * adv
  policy_loss         ()     scalar, mean over N
  entropy             ()     scalar, -mean(new_logp) or 0
  approx_kl           ()     scalar, mean(old_logp - new_logp), logging only
  kl_ref              ()     scalar, the KL penalty (kl_mean)
  total_loss          ()     scalar fed to .backward()  (policy_loss - ent*entropy + kl_coef*kl_ref)
"""
from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class PolicyOnlyLossOut:
    # Bundle returned to train_grpo.py so it can backprop `total_loss` and log the pieces.
    # NOTE: no value_loss field (unlike Part 8's PPOLossOut) -- GRPO has no critic.
    policy_loss: torch.Tensor   # () clipped-surrogate loss (drives the actor)
    entropy: torch.Tensor       # () approx entropy bonus term (off by default, ent_coef=0)
    approx_kl: torch.Tensor     # () mean(old_logp - new_logp), DIAGNOSTIC only
    kl_ref: torch.Tensor        # () KL(new||ref) penalty term that enters total
    total_loss: torch.Tensor    # () policy_loss - ent_coef*entropy + kl_coef*kl_ref


def ppo_policy_only_losses(new_logp, old_logp, adv, clip_ratio=0.2, ent_coef=0.0,
                           kl_coef: float = 0.0, kl_mean: torch.Tensor | None = None):
    """
    PPO-style clipped policy loss, *policy only* (no value head),
    plus a separate KL(π||π_ref) penalty term:  total = L_PPO + kl_coef * KL.
    Inputs are flat over action tokens: new_logp, old_logp, adv: (N_act,).
    kl_mean is a scalar tensor (mean over action tokens).
    """
    device = new_logp.device if new_logp.is_cuda else None
    if new_logp.numel() == 0:                # no action tokens this batch -> all-zero losses
        zero = torch.tensor(0.0, device=device)
        return PolicyOnlyLossOut(zero, zero, zero, zero, zero)

    ratio = torch.exp(new_logp - old_logp)  # (N,) pi_new/pi_old per token; ==1 before any update
    unclipped = ratio * adv                 # (N,) raw surrogate
    # (N,) clamp ratio to [1-clip, 1+clip] (default [0.8, 1.2]) so one update can't drift too far
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    # min() = pessimistic branch -> flattens objective past the clip band; minus sign: we maximize.
    policy_loss = -torch.mean(torch.min(unclipped, clipped))  # () SAME clipped surrogate as PPO

    # entropy bonus is OFF by default (ent_coef=0). When on, -mean(new_logp) is a cheap proxy.
    entropy = -new_logp.mean() if ent_coef != 0.0 else new_logp.new_tensor(0.0)  # ()
    approx_kl = torch.mean(old_logp - new_logp)  # () drift-from-snapshot diagnostic (not in the loss)

    kl_ref = kl_mean if kl_mean is not None else new_logp.new_tensor(0.0)  # () KL(new||ref) penalty

    # () actor + KL penalty; NO value loss (no critic). entropy bonus was not used in original GRPO paper
    total = policy_loss - ent_coef * entropy + kl_coef * kl_ref
    return PolicyOnlyLossOut(policy_loss, entropy, approx_kl, kl_ref, total)
