"""9.2 Rollout — let the policy "do a take" and record per-token logprobs (called G times per prompt).

What this file does
-------------------
Identical to Part 8's rollout toolbox -- GRPO reuses it verbatim. The only thing
that changes is *upstream*: train_grpo.py calls `generate` G times per prompt to
build a GROUP of takes (Part 8 generated one). Each take still needs its per-token
log-probabilities recorded ("old_logp"), so GRPO can later form the clipped ratio.
This file supplies:

  - RLHFTokenizer    : turn text <-> token ids (BPE from Part 4, else ByteTokenizer
                       from Part 3). _HAS_BPE records which one we got.
  - shift_labels     : line up "what the model predicted" with "what came next".
  - gather_logprobs  : pluck out log p(actual token) for every position.
  - model_logprobs   : run the model (no_grad) and return per-token logprobs of a
                       sequence -- used to snapshot old_logp and the ref logp.
  - approx_kl        : a cheap one-number estimate of how far policy drifted from ref.
  - sample_prompts   : grab a few real prompts (alpaca) or fall back to 4 canned ones.

Tiny mental model of the logprob alignment (per take)::

    x        = [ x0   x1   x2   x3 ]          # a token sequence, length T
    logits   = model(x)                       # (T,V): row t predicts token t+1
    use rows  0    1    2                      # logits[:, :-1]  (drop last row)
    vs labels  x1   x2   x3                    # shift_labels(x) (drop first token)
    logp[t]  = log p(x[t+1] | x[:t+1])         # length T-1

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
Causal-LM per-token logprob (the alignment that shift_labels + gather_logprobs do):

    logits      = model(x)                       # (B,T,V); row t scores token t+1
    p(token)    = softmax(logits, dim=-1)        # over the vocab V
    logp[b,t]   = log p( x[b, t+1] | x[b, :t+1] ) = log_softmax(logits)[b, t, x[b,t+1]]

    We use logits[:, :-1, :] (rows 0..T-2 predict positions 1..T-1) against
    labels = x[:, 1:] (the tokens those rows are predicting). Result is length T-1.
      B = batch size, T = sequence length, V = vocab size.

Approx KL between policy and frozen reference (a leash, not an exact KL):

    approx_kl = mean_over_tokens( logp_policy - logp_ref )

    If the policy has not moved, logp_policy == logp_ref and approx_kl == 0.
    Larger value => policy is drifting away from the frozen SFT reference.

Visualization
-------------
See notebook section 9.2 -- generating a GROUP of G takes for one prompt and recording
each take's per-token logprobs (the same recording Part 8 did for a single take).

Shapes
------
  x                          (B, T)         a batch of token-id sequences
  logits = model(x)          (B, T, V)      one score per vocab token, per position
  shift_labels(x)            (B, T-1)       x[:, 1:], the "next token" targets
  logits[:, :-1, :]          (B, T-1, V)    rows that predict positions 1..T-1
  gather_logprobs(...)       (B, T-1)       log p(actual next token) per position
  model_logprobs(model, x)   (B, T-1)       no_grad per-token logprobs of x
  approx_kl(pi, ref)         ()             a single scalar (mean over all tokens)
  sample_prompts(n)          list[str]      n prompt strings (alpaca or fallback)
"""
from __future__ import annotations
import torch
from typing import List, Tuple

# tokenizer pref: BPE from Part 4 → fallback to ByteTokenizer from Part 3
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))
try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

from part_6.formatters import Example, format_example, format_prompt_only

# ---------- tokenizer helpers ----------
class RLHFTokenizer:
    def __init__(self, block_size: int, bpe_dir: str | None = None, vocab_size: int = 8000):
        self.block_size = block_size
        self.tok = None
        if _HAS_BPE:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size)
                if bpe_dir:
                    self.tok.load(bpe_dir)
            except Exception:
                self.tok = None
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        if self.tok is None:
            raise RuntimeError("No tokenizer available for RLHF.")

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    def decode(self, ids: List[int]) -> str:
        if hasattr(self.tok, 'decode'):
            return self.tok.decode(ids)
        return bytes(ids).decode('utf-8', errors='ignore')

# ---------- logprob utilities ----------

def shift_labels(x: torch.Tensor) -> torch.Tensor:
    # For causal LM: predict x[t+1] from x[:t]
    return x[:, 1:].contiguous()  # (B,T) -> (B,T-1): drop first token (no row predicts it)

def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs of the given labels.
    logits: (B,T,V), labels: (B,T) over same T
    returns: (B,T) log p(labels)
    """
    logp = torch.log_softmax(logits, dim=-1)            # (B,T,V): log-prob over vocab per position
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # pick label's logp -> (B,T)

@torch.no_grad()
def model_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    # compute log p(x[t+1] | x[:t]) for t
    logits, _, _ = model.lm(x, None) if hasattr(model, 'lm') else model(x, None)  # logits: (B,T,V)
    labels = shift_labels(x)                       # (B,T-1): the next-token targets
    lp = gather_logprobs(logits[:, :-1, :], labels)  # drop last row -> gathered to (B,T-1)
    return lp  # (B, T-1)

# ---------- KL ----------

def approx_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    # Mean over tokens: KL(pi||ref) ≈ (logp_pi - logp_ref).mean()
    return (policy_logp - ref_logp).mean()  # (B,T-1) - (B,T-1) -> mean -> () scalar

# ---------- small prompt source ----------
try:
    from datasets import load_dataset as _load_ds
except Exception:
    _load_ds = None

def sample_prompts(n: int) -> List[str]:
    if _load_ds is not None:
        try:
            ds = _load_ds("tatsu-lab/alpaca", split="train[:24]")
            arr = []
            for r in ds:
                inst = (r.get('instruction') or '').strip()
                inp = (r.get('input') or '').strip()
                if inp:
                    inst = inst + "\n" + inp
                if inst:
                    arr.append(inst)
                if len(arr) >= n:
                    break
            if arr:
                return arr
        except Exception:
            pass
    # fallback
    base = [
        "Explain the purpose of attention in transformers.",
        "Give two pros and cons of BPE tokenization.",
        "Summarize why PPO is used in RLHF.",
        "Write a tiny Python function that reverses a list.",
    ]
    return (base * ((n+len(base)-1)//len(base)))[:n]