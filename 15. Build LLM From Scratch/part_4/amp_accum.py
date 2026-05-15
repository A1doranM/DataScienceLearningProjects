"""4.3 AMP (mixed precision) + gradient accumulation, in one wrapper.

What this file does
-------------------
`AmpGrad` wraps an optimizer with two orthogonal scaling tricks:

  1. **Mixed precision (AMP)** via `torch.cuda.amp.GradScaler`:
       * forward runs in fp16/bf16 (under `autocast` in the train loop)
       * loss is multiplied by a dynamic `scale` before backward to keep
         gradients above fp16's underflow region
       * `scaler.step(opt)` unscales gradients, skips the step if any
         non-finite values appeared, then `scaler.update()` adjusts scale

  2. **Gradient accumulation** over `accum` micro-batches:
       * each `backward()` divides the loss by `accum` (so the running
         sum stays the same scale as one large batch)
       * `should_step()` returns True every `accum` calls
       * caller decides when to `step()` + `zero_grad()`

Where this fits in the training pipeline
----------------------------------------
    [ forward: GPTModern  ]
              |
    [ CE loss             ]
              |
    [ 4.3 AMP + grad accum] <-- THIS FILE
              |
    [ AdamW.step()        ]   (only when `should_step()`)
              ...

Why AMP?
--------
fp16 is ~2x faster than fp32 on every NVIDIA GPU since Volta (V100) and
uses half the memory. The catch: fp16's exponent range is small, so
small gradients underflow to zero. `GradScaler` solves it by scaling the
loss up before backward so gradients live in a safe range, then unscaling
before the optimizer step.

Why gradient accumulation?
--------------------------
"Effective batch size = micro_batch_size * accum_steps." When a target
batch size doesn't fit in VRAM, run several micro-batches and accumulate
their gradients without zeroing in between. Mathematically identical to
one big batch (up to BN/dropout stochasticity).

Caller pattern
--------------
    amp = AmpGrad(optimizer, accum=4, amp=True)
    with torch.cuda.amp.autocast(enabled=amp.amp):
        logits, loss, _ = model(xb, yb)
    amp.backward(loss)            # always scales+accumulates
    if amp.should_step():
        amp.step()                # scaler.step + scaler.update
        amp.zero_grad()
        lr = scheduler.step()

Shapes
------
  loss (input)                  : scalar tensor
  scaler.state_dict() returned  : small dict (saved in checkpoint)
"""
import torch

class AmpGrad:
    """AMP + gradient accumulation wrapper.
    Usage:
        amp = AmpGrad(optimizer, accum=4, amp=True)
        amp.backward(loss)
        if amp.should_step(): amp.step(); amp.zero_grad()
    """
    def __init__(self, optimizer, accum: int = 1, amp: bool = True):
        self.optim = optimizer
        self.accum = max(1, accum)
        self.amp = amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self._n = 0
    def backward(self, loss: torch.Tensor):
        loss = loss / self.accum
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._n += 1
    def should_step(self):
        return (self._n % self.accum) == 0
    def step(self):
        if self.amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()
    def zero_grad(self):
        self.optim.zero_grad(set_to_none=True)