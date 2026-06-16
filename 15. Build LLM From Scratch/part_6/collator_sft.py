"""6.3 SFTCollator — tokenize an (instruction, response) pair and MASK the prompt.

What this file does
-------------------
This is the heart of Part 6. Supervised fine-tuning (SFT) reuses the *same*
next-token training objective as the base model from Part 4, but with one twist:
we do NOT want the model to be graded on predicting the question — only on
predicting the answer. So for every example we:

  1. build the full "prompt + response" string with the Part 6 chat template,
  2. tokenize it into `ids`,
  3. separately tokenize the prompt-only text to learn how long the prompt is,
  4. build labels `y` as the usual causal shift of `ids`, and finally
  5. overwrite every prompt-position label with -100 so the loss IGNORES it.

In pseudo-code the label construction is::

    y = ids.copy()
    for t in range(len(y) - 1):   # causal shift
        y[t] = ids[t + 1]
    y[-1] = -100                  # nothing comes after the last token
    for i in range(n_prompt - 1): # mask the prompt question
        y[i] = -100

The label value -100 is PyTorch's `ignore_index` for cross-entropy, so those
positions contribute zero gradient. Everything else (the response tokens) is
trained normally. Finally each sequence is padded to `block_size`: inputs with
pad id `2`, labels with `-100` so padding is ignored too.

Where this fits in the Part 6 SFT pipeline
------------------------------------------
    raw (instruction, response) pair
              |
    [ 6.1 chat template            (formatters.py)   ]
              |
    [ 6.2 dataset of pairs         (dataset_sft.py)  ]
              |
    [ 6.3 tokenize + MASK prompt   (collator_sft.py) ]   <-- THIS FILE   labels = -100 on the question
              |
    [ load Part 4 base checkpoint                    ]
    [ 6.4 curriculum: easy -> hard (curriculum.py)   ]
              |
    [ 6.5 masked-CE training loop  (train_sft.py)    ]
              |
    [ 6.6 sample with the template (sample_sft.py)   ]
              |
    [ 6.7 evaluate EM / token-F1   (evaluate.py)     ]

Math / Logic (label construction for one example)
-------------------------------------------------
  Let `ids` = token ids of "<prompt template><response>", length L (<= block_size).
  Let `n_prompt` = number of tokens that belong to the prompt-only prefix.

    causal shift : y[t] = ids[t+1]   for t = 0 .. L-2
                          (each position predicts the NEXT token)
    last token   : y[L-1] = -100     (no token follows the final one)
    mask prompt  : y[i]   = -100     for i = 0 .. n_prompt-2

  Why `n_prompt - 1` and not `n_prompt`?  Position i predicts token i+1.
  The prompt occupies positions 0 .. n_prompt-1. Position `n_prompt-1` predicts
  token `n_prompt`, which is the FIRST RESPONSE TOKEN — that prediction is the
  whole point of fine-tuning, so it is graded (NOT masked). We therefore only
  blank out positions 0 .. n_prompt-2, i.e. `range(n_prompt - 1)`.

Visualization
-------------
See notebook section 6.3 — shows a real prompt/response pair side by side with
its labels, with the prompt region greyed out (-100) and the response region
highlighted as the only tokens that count toward the loss.

Shapes
------
  one example:
    ids / x      : list[int] length L  (L = min(len(full ids), block_size))
    y (labels)   : list[int] length L  (causal-shifted, -100 on prompt + last)
    n_prompt     : int                 (= min(len(prompt_ids), len(ids)))
  after padding (a batch of B examples):
    x  (input_ids) : (B, block_size) long  pad id = 2
    y  (labels)    : (B, block_size) long  pad id = -100 (ignored by CE)
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import traceback

# Reuse tokenizers: prefer BPE from Part 4 if available; else byte-level from Part 3
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

from formatters import Example, format_example, format_prompt_only

class SFTCollator:
    """Turn (instruction,response) into token ids and masked labels for causal LM (6.2).
    Labels for the prompt part are set to -100 so they don't contribute to loss.
    """
    def __init__(self, block_size: int = 256, bpe_dir: str | None = None):
        self.block_size = block_size
        self.tok = None
        if _HAS_BPE:
            # If a trained tokenizer directory exists from Part 4, you can `load` it.
            # Otherwise we create an ad-hoc BPE on the fly using fallback prompts during demo.
            try:
                self.tok = BPETokenizer(vocab_size=8000)
                if bpe_dir:
                    self.tok.load(bpe_dir)
                    print(f"Loaded BPE tokenizer from {bpe_dir}")
                else:
                    # weak ad-hoc training would belong elsewhere; for the demo we assume Part 4 tokenizer exists
                    pass
            except Exception:
                print(traceback.format_exc())
                self.tok = None
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        if self.tok is None:
            raise RuntimeError("No tokenizer available. Install tokenizers or ensure Part 3 ByteTokenizer exists.")

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    def encode(self, text: str) -> List[int]:
        if hasattr(self.tok, 'encode'):
            ids = self.tok.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        # ByteTokenizer-like
        return list(text.encode('utf-8'))

    def collate(self, batch: List[Tuple[str,str]]):
        # Build "prompt + response" and create label mask where prompt positions are -100.
        input_ids = []
        labels = []
        for prompt, response in batch:
            prefix_text = format_prompt_only(prompt).replace('</s>','')   # prompt-only text (drop end marker)
            text = format_example(Example(prompt, response))              # full "prompt + response" text
            ids = self.encode(text)[:self.block_size]                     # full token ids, length L <= block_size
            prompt_ids = self.encode(prefix_text)[:self.block_size]       # prompt-only token ids
            n_prompt = min(len(prompt_ids), len(ids))                     # how many leading tokens are the prompt
            x = ids                                                       # input ids: list[int] length L
            y = ids.copy()                                                # labels start as a copy of ids
            for t in range(len(y) - 1):
                y[t] = ids[t + 1]                                         # causal shift: position t predicts token t+1
            y[-1] = -100                                                  # last token has no "next" -> ignore in loss
            for i in range(n_prompt-1):
                y[i] = -100                                               # mask the prompt; pos n_prompt-1 (1st response token) stays graded
            input_ids.append(x)
            labels.append(y)
        # pad to block_size
        def pad_to(ids, val):
            if len(ids) < self.block_size:
                ids = ids + [val]*(self.block_size - len(ids))
            return ids[:self.block_size]
        x = torch.tensor([pad_to(s, 2) for s in input_ids], dtype=torch.long)   # (B, block_size) inputs, pad id = 2
        y = torch.tensor([pad_to(s, -100) for s in labels], dtype=torch.long)   # (B, block_size) labels, pad = -100 (ignored)
        return x, y