"""7.2 PairCollator — turn preference pairs into padded (pos, neg) token ids.

What this file does
-------------------
A preference example is a triple (prompt, chosen, rejected): the same question
plus a *better* answer and a *worse* answer. Before the reward model can score
them, both answers must become token ids that look exactly alike in format. This
file is that plating + tokenizing step — text in, two id tensors out, no
learnable weights.

For each pair it does the same thing twice (once for chosen, once for rejected):

    pos_text = format_example(Example(prompt, chosen))     # Part 6 SFT template
    neg_text = format_example(Example(prompt, rejected))   # same template
    ids      = encode(text)[:block_size]                   # tokenize, then clip
    ids      = ids + [2, 2, ...]  until len == block_size  # pad with id 2

Plating reuses Part 6's ``format_example`` so every answer is wrapped in the
identical "### Instruction: ... ### Response: ..." string — the judge sees one
uniform format. Tokenizing prefers the Part 4 BPE tokenizer; if it is not
available it falls back to the Part 3 ByteTokenizer (one byte = one id). Pad id
2 matches ``model_reward.py``, which masks ``(x == 2)`` so pads are ignored in
the attention and the mean-pool.

Where this fits in the Part 7 reward-modeling pipeline
------------------------------------------------------
    preference pair (prompt, chosen, rejected)        (data_prefs.py)
              |
    [ plate both with SFT template + tokenize         (collator_rm.py)  ]   <-- THIS FILE
    [   -> (pos_ids, neg_ids)                                           ]
              |
    [ reward model: encoder -> mean-pool -> scalar    (model_reward.py) ]
    [   r_pos = score(chosen),  r_neg = score(rejected)                 ]
              |
    [ Bradley-Terry loss on the gap                   (loss_reward.py)  ]
    [   softplus(-(r_pos - r_neg))                                      ]
              |
    [ train -> reward checkpoint                      (train_rm.py)     ]
              |
    [ eval: pairwise accuracy r_pos > r_neg           (eval_rm.py)      ]

Visualization
-------------
See notebook section 7.2 — shows a (prompt, chosen, rejected) pair being plated
into the SFT template and tokenized into chosen-ids / rejected-ids, ready for
the judge to score.

Shapes
------
  batch              : List[(prompt, chosen, rejected)]  length B   (strings)
  pos_text, neg_text : str (one plated answer each)
  per-row ids        : List[int], clipped to <= block_size, then padded
  pos                : (B, block_size)  long  (chosen answers, pad id = 2)
  neg                : (B, block_size)  long  (rejected answers, pad id = 2)
"""
from __future__ import annotations
from typing import List, Tuple
import torch

# Prefer BPE from Part 4, else ByteTokenizer from Part 3
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

sys.path.append(str(_P(__file__).resolve().parents[1]/'part_6'))
try:
    from formatters import Example, format_example  # reuse formatting
except Exception:
    pass

class PairCollator:
    """Tokenize preference pairs into (pos, neg) input ids.
    We format as the SFT template with the 'chosen' or 'rejected' text as the Response.
    """
    def __init__(self, block_size: int = 256, bpe_dir: str | None = None, vocab_size: int | None = None):
        self.block_size = block_size
        self.tok = None
        if _HAS_BPE:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size or 8000)
                if bpe_dir:
                    self.tok.load(bpe_dir)
            except Exception:
                self.tok = None
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        if self.tok is None:
            raise RuntimeError("No tokenizer available.")

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tok, 'encode'):
            ids = self.tok.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        return list(text.encode('utf-8'))

    def collate(self, batch: List[Tuple[str, str, str]]):
        # batch of (prompt, chosen, rejected) strings, length B
        pos_ids, neg_ids = [], []  # per-row List[int], one entry per pair
        for prompt, chosen, rejected in batch:
            pos_text = format_example(Example(prompt, chosen))    # str: chosen plated in SFT template
            neg_text = format_example(Example(prompt, rejected))  # str: rejected plated in same template
            pos_ids.append(self._encode(pos_text)[:self.block_size])  # ids clipped to <= block_size
            neg_ids.append(self._encode(neg_text)[:self.block_size])  # ids clipped to <= block_size
        def pad_to(x, pad=2):
            # right-pad short rows with id 2 (model masks x == 2), or hard-clip long ones -> len == block_size
            return x + [pad] * (self.block_size - len(x)) if len(x) < self.block_size else x[:self.block_size]
        pos = torch.tensor([pad_to(x) for x in pos_ids], dtype=torch.long)  # (B, block_size) chosen
        neg = torch.tensor([pad_to(x) for x in neg_ids], dtype=torch.long)  # (B, block_size) rejected
        return pos, neg  # (B, block_size), (B, block_size)