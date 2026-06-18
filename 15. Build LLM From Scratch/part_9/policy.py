"""9.1 PolicyWithValue — the actor (the value-head critic is along for the ride, unused).

What this file does
-------------------
This is the SAME class as Part 8: it bolts a value head onto our Part 3 language
model. But GRPO is *actor-only* — it never reads the value head. We keep the class
so Part 8 and Part 9 can share `policy.py` and the same checkpoints; the GRPO loop
just discards the `values` output:

    PolicyWithValue
      |
      +-- self.lm       = GPTModern  (the SFT LM -> the ACTOR, picks tokens)       <-- GRPO uses this
      |
      +-- self.val_head = Linear(V, 1)  (the CRITIC)                               <-- GRPO ignores this

In Part 8 the value head was the PPO baseline ("how good is this position?").
GRPO replaces that baseline with the *group average reward* (see train_grpo.py /
notebook 9.4), so the critic becomes dead weight. When you read
`logits_new, _, _ = policy(seq, None)` in train_grpo.py, that discarded middle
slot is exactly the fired critic.

Where this fits in the Part 9 RLHF-GRPO loop
--------------------------------------------
    prompt
       |
    [ policy does a GROUP of takes: G generations + old_logp  (policy.py / rollout.py) ]   <-- THIS FILE
       |
    [ judge scores each take: reward r_1..r_G                 (part_7 RewardModel)     ]
       |
    [ group baseline: A_i = r_i - mean(r over the group)      (train_grpo.py)          ]
       |
    [ clipped surrogate: min(unclipped, clipped)             (grpo_loss.py)           ]
       |
    [ + KL penalty in the loss: kl_coef * KL(new || ref)     (grpo_loss.py)           ]
       |
    [ AdamW step on the policy only (no value head)          (train_grpo.py)          ]
       |
    [ eval: avg reward, tuned policy vs frozen ref           (eval_ppo.py)            ]

Math
----
The value head is just a linear projection of the logits, with bias=False:

    values[b,t] = sum_v  logits[b,t,v] * W[v]

  logits[b,t,v] = LM score for token id v at position t in sequence b
  W[v]          = the single weight column of val_head (one number per vocab id)
  values[b,t]   = scalar baseline -- COMPUTED on every forward, but GRPO never uses it

(The GRPO clipped surrogate, group baseline, and KL penalty live in the sibling
files named in the diagram above -- this file only supplies the logits the actor
needs; the group average, not this head, is GRPO's baseline.)

Visualization
-------------
See notebook section 9.1 — the actor-critic split with the critic greyed out:
the LM trunk feeds a token-picking head (used) and a one-number value head (ignored).

Shapes
------
  x            (B, T)        token ids fed to the LM
  logits       (B, T, V)     per-token vocabulary scores from GPTModern  (GRPO uses these)
  val_head     (V) -> (1)    Linear maps the V logits to one scalar
  values       (B, T)        squeeze of (B, T, 1); computed but DISCARDED by GRPO
  loss         scalar | None LM cross-entropy when targets y are passed, else None
    B = batch, T = sequence length, V = vocab_size

Parameter count
---------------
  val_head : vocab_size * 1 = vocab_size weights (bias=False) -- now dead weight under GRPO
  self.lm  : the full GPTModern (n_layer x transformer blocks + embeddings),
             millions of params -- the LM is the whole policy; the value head is unused
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
        self.lm = GPTModern(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                            n_head=n_head, n_embd=n_embd, use_rmsnorm=use_rmsnorm,
                            use_swiglu=use_swiglu, rope=rope, dropout=dropout)
        # value head over logits (toy). Shapes: (B,T,V) -> (B,T,1) -> (B,T)
        self.val_head = nn.Linear(vocab_size, 1, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # Delegate LM forward; returns logits (B,T,V), loss, _
        logits, loss, _ = self.lm(x, y)
        values = self.val_head(logits).squeeze(-1)  # (B,T,V)->(B,T,1)->(B,T); GRPO discards this
        return logits, values, loss  # GRPO calls `logits, _, _ = policy(...)` -> drops `values`

    def generate(self, *args, **kwargs):
        # sampling is pure-actor: forward straight to the LM (value head not involved)
        return self.lm.generate(*args, **kwargs)