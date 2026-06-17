"""8.1 PolicyWithValue — the actor-critic wrapper PPO actually optimizes.

What this file does
-------------------
PPO needs two things from one network: a *policy* (what token to say next) and a
*value* (how good is this spot, used as a baseline so the advantage isn't noisy).
This file bolts both onto our existing Part 3 language model:

    PolicyWithValue
      |
      +-- self.lm       = GPTModern  (the SFT LM -> the ACTOR, picks tokens)
      |
      +-- self.val_head = Linear(V, 1)  (the CRITIC, scores each position)

On a forward pass the LM produces per-token logits, the value head squeezes
those logits down to one scalar per position, and we hand back
(logits, values, loss). NOTE the value head reads the *logits* (V numbers per
token), not the hidden state -- a deliberately toy choice the class docstring
flags, so the tutorial stays runnable without poking at GPTModern internals.

Where this fits in the Part 8 RLHF-PPO loop
-------------------------------------------
    prompt
       |
    [ policy does a take: generate + record old_logp   (policy.py / rollout.py) ]   <-- THIS FILE
       |
    [ judge scores the take: reward model r            (part_7 RewardModel)     ]
       |
    [ KL leash: shaped_r = r - kl * (logp - ref_logp)  (vs frozen SFT reference)]
       |
    [ advantage = shaped_r - value baseline, normalized (train_ppo.py)          ]
       |
    [ PPO clipped update: min(unclipped, clipped)      (ppo_loss.py)            ]
       |
    [ AdamW step on the policy only                    (train_ppo.py)           ]
       |
    [ eval: avg reward, tuned policy vs frozen ref     (eval_ppo.py)            ]

Math
----
The value head is just a linear projection of the logits, with bias=False:

    values[b,t] = sum_v  logits[b,t,v] * W[v]

  logits[b,t,v] = LM score for token id v at position t in sequence b
  W[v]          = the single weight column of val_head (one number per vocab id)
  values[b,t]   = scalar "how good is this position" baseline for PPO

(The actual PPO clipped surrogate, advantage, and KL leash live in the
sibling files named in the diagram above -- this file only supplies the
logits + value baseline they consume.)

Visualization
-------------
See notebook section 8.1 — actor-critic split: one shared LM trunk feeding a
token-picking head (policy) and a one-number value head (critic baseline).

Shapes
------
  x            (B, T)        token ids fed to the LM
  logits       (B, T, V)     per-token vocabulary scores from GPTModern
  val_head     (V) -> (1)    Linear maps the V logits to one scalar
  values       (B, T)        squeeze of (B, T, 1); one baseline per position
  loss         scalar | None LM cross-entropy when targets y are passed, else None
    B = batch, T = sequence length, V = vocab_size

Parameter count
---------------
  val_head : vocab_size * 1 = vocab_size weights (bias=False), e.g. ~50k floats
  self.lm  : the full GPTModern (n_layer x transformer blocks + embeddings),
             millions of params -- the LM dominates; the value head is a rounding error
"""
from __future__ import annotations
import torch, torch.nn as nn
import sys
from pathlib import Path as _P
# Try user’s structure first
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from model_utils.model_modern import GPTModern  # user-custom path
except Exception:
    from model_modern import GPTModern  # fallback

class PolicyWithValue(nn.Module):
    """Policy network = SFT LM + tiny value head.
    NOTE: For simplicity we place value head on top of LM logits (vocab→1).
    This avoids depending on hidden-state internals while keeping the tutorial runnable.
    """
    def __init__(self, vocab_size: int, block_size: int, n_layer=4, n_head=4, n_embd=256,
                 use_rmsnorm=True, use_swiglu=True, rope=True, dropout=0.0):
        super().__init__()
        # the ACTOR: reuses the Part 3 LM; forward(x) -> logits (B,T,V)
        self.lm = GPTModern(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                            n_head=n_head, n_embd=n_embd, use_rmsnorm=use_rmsnorm,
                            use_swiglu=use_swiglu, rope=rope, dropout=dropout)
        # value head over logits (toy). Shapes: (B,T,V) -> (B,T,1) -> (B,T)
        self.val_head = nn.Linear(vocab_size, 1, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # Delegate LM forward; returns logits (B,T,V), loss, _
        logits, loss, _ = self.lm(x, y)
        values = self.val_head(logits).squeeze(-1)  # (B,T,V) -> (B,T,1) -> (B,T)
        return logits, values, loss  # logits (B,T,V), values (B,T), loss scalar|None

    def generate(self, *args, **kwargs):
        # sampling is pure-actor: forward straight to the LM, value head not involved
        return self.lm.generate(*args, **kwargs)