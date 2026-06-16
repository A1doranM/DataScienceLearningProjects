"""6.2 SFT dataset — load tiny (instruction, response) pairs for supervised fine-tuning.

What this file does
-------------------
Builds the list of training examples for SFT (supervised fine-tuning). Each
example is just a `(prompt, response)` pair wrapped in a tiny `SFTItem` dataclass.
We try to download a small slice of a real instruction dataset (Alpaca) and, if
that is unavailable (no internet, `datasets` not installed, or `sample_dataset=True`),
we fall back to a 3-item baked-in list so the rest of Part 6 always has data to run on.

Alpaca rows have three text fields; we merge `instruction` + `input` into one
prompt and use `output` as the response::

    instr = instruction.strip()
    inp   = input.strip()
    out   = output.strip()
    if inp:                       # some rows add extra context in `input`
        instr = instr + "\\n" + inp
    if instr and out:             # skip empty rows (missing prompt OR answer)
        items.append(SFTItem(prompt=instr, response=out))

The result is a plain Python list `List[SFTItem]` — no tensors yet. Turning these
strings into the chat template happens in 6.1 (formatters.py) and tokenizing +
masking the prompt happens next in 6.3 (collator_sft.py).

Where this fits in the Part 6 SFT pipeline
------------------------------------------
    raw (instruction, response) pair
              |
    [ 6.1 chat template            (formatters.py)   ]
              |
    [ 6.2 dataset of pairs         (dataset_sft.py)  ]   <-- THIS FILE
              |
    [ 6.3 tokenize + MASK prompt   (collator_sft.py) ]   labels = -100 on the question
              |
    [ load Part 4 base checkpoint                    ]
    [ 6.4 curriculum: easy -> hard (curriculum.py)   ]
              |
    [ 6.5 masked-CE training loop  (train_sft.py)    ]
              |
    [ 6.6 sample with the template (sample_sft.py)   ]
              |
    [ 6.7 evaluate EM / token-F1   (evaluate.py)     ]

Visualization
-------------
See notebook section 6.2 — how a few raw Alpaca rows turn into clean
(prompt, response) pairs, and what the 3-item fallback list looks like.

Shapes
------
  load_tiny_hf(...) returns : List[SFTItem]          one item per usable example
  SFTItem.prompt            : str                    instruction (+ optional input merged in)
  SFTItem.response          : str                    the target answer text
  Alpaca slice "train[:200]": up to 200 rows         minus any empty-row skips
  fallback list             : exactly 3 SFTItem      used when HF load is unavailable
"""
from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
import traceback

try:
    from datasets import load_dataset
except Exception:
    print("Couldn't import `datasets`. Will use fallback data only.")
    load_dataset = None

from formatters import Example

@dataclass
class SFTItem:
    prompt: str      # the instruction text the model reads (its "question")
    response: str    # the target answer text the model must learn to produce


def load_tiny_hf(split: str = "train[:200]", sample_dataset: bool = False) -> List[SFTItem]:
    """Try to load a tiny instruction dataset from HF; fall back to a baked-in list.
    We use `tatsu-lab/alpaca` as a familiar schema (instruction, input, output) and keep only a slice.
    """
    items: List[SFTItem] = []                        # accumulates one SFTItem per usable row
    # Path A: try the real Alpaca slice (skipped if `datasets` missing or sample_dataset=True)
    if load_dataset is not None and not sample_dataset:
        try:
            ds = load_dataset("tatsu-lab/alpaca", split=split)  # e.g. up to 200 rows
            for row in ds:
                instr = row.get("instruction", "").strip()  # str: the task to do
                inp = row.get("input", "").strip()          # str: optional extra context
                out = row.get("output", "").strip()         # str: the reference answer
                if inp:
                    instr = instr + "\n" + inp               # merge input into the prompt
                if instr and out:                            # empty-row skip: need both halves
                    items.append(SFTItem(prompt=instr, response=out))
        except Exception:
            pass
    if not items:
        # fallback tiny list (3 items) — used when the HF load above produced nothing
        seeds = [
            ("First prime number", "2"),
            ("What are the three primary colors?", "red"),
            ("Device name which points to direction?", "compass"),
        ]
        items = [SFTItem(prompt=p, response=r) for p,r in seeds]  # 3 SFTItem
    return items                                     # -> List[SFTItem]