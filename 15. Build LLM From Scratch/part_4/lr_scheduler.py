"""4.2 Warmup + Cosine learning rate schedule (per-step API).

What this file does
-------------------
Implements the canonical LLM learning rate schedule:

  1. **Linear warmup** for the first `warmup_steps` optimizer steps:
       lr(step) = base_lr * step / warmup_steps

  2. **Cosine decay** to 0 over the remaining steps:
       progress = (step - warmup) / (total - warmup)
       lr(step) = 0.5 * base_lr * (1 + cos(pi * progress))

`.step()` returns the new learning rate and writes it into every
`param_group['lr']` of the wrapped optimizer.

Where this fits in the training pipeline
----------------------------------------
    [ ...                          ]
              |
    [ AdamW.step()                 ]
              |
    [ 4.2 Warmup + Cosine LR       ]   <-- THIS FILE
              |
    [ 4.5 Logger / 4.4 Checkpoint  ]
              |
    [ next batch                   ]

Why warmup?
-----------
At init the model's weights are random and Q/K dot products are wild.
Stepping at full learning rate immediately can produce huge gradients
that damage the careful Xavier/Kaiming init or cause loss spikes.
Linear warmup eases the model into useful gradients.

Why cosine decay (vs step / linear / exponential)?
--------------------------------------------------
Smooth, monotone decrease ending exactly at 0; no hyperparameters
beyond `total_steps`. Empirically beats step decay for transformer
training and is what every Llama / GPT-NeoX / Phi recipe uses.

Math
----
    if step <= warmup:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = 0.5 * base_lr * (1 + cos(pi * progress))

Visualization
-------------
See notebook section 4.2 — full curve plotted with shaded warmup region.
"""
import math

class WarmupCosineLR:
    """Linear warmup → cosine decay (per-step API)."""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps+1, total_steps)
        self.base_lr = base_lr
        self.step_num = 0
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr