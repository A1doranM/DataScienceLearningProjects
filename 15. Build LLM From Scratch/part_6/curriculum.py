"""6.4 LengthCurriculum — feed examples easy -> hard (short prompts first).

What this file does
-------------------
Wraps a list of (prompt, response) string pairs and hands them back in a
fixed order: shortest prompt first, longest prompt last. "Easy -> hard"
here is a simple proxy — we assume a short instruction is an easier thing
for the model to learn than a long one, so we let it warm up on the easy
pairs before facing the hard ones. The whole thing is one tiny class:

    self.items = sorted(items, key=len(prompt))   # short prompt -> long prompt
    iterating yields items[0], items[1], ... once, then StopIteration

It is a *single-pass* iterator: __iter__ resets a cursor to 0 and __next__
walks it forward one item at a time. To do another epoch you call iter()
on the object again (which re-zeros the cursor). Note the sort key is
len(p[0]) = the number of *characters* in the prompt string, measured
before tokenization — a cheap stand-in for "true" difficulty.

Where this fits in the Part 6 SFT pipeline
------------------------------------------
    raw (instruction, response) pair
              |
    [ 6.1 chat template            (formatters.py)   ]
              |
    [ 6.2 dataset of pairs         (dataset_sft.py)  ]
              |
    [ 6.3 tokenize + MASK prompt   (collator_sft.py) ]   labels = -100 on the question
              |
    [ load Part 4 base checkpoint                    ]
    [ 6.4 curriculum: easy -> hard (curriculum.py)   ]   <-- THIS FILE
              |
    [ 6.5 masked-CE training loop  (train_sft.py)    ]
              |
    [ 6.6 sample with the template (sample_sft.py)   ]
              |
    [ 6.7 evaluate EM / token-F1   (evaluate.py)     ]

Visualization
-------------
See notebook section 6.4 — shows the (prompt, response) pairs reordered
from shortest to longest prompt, illustrating the easy-to-hard schedule.

Shapes
------
  input  items     : list of N tuples (prompt: str, response: str)
  after sort        : same N tuples, reordered by len(prompt) ascending
  each __next__     : one tuple (prompt: str, response: str)
  total yielded     : N items per pass, then StopIteration
"""
from __future__ import annotations
from typing import List

class LengthCurriculum:
    """6.3 Curriculum: iterate examples from short→long prompts (one pass demo)."""
    def __init__(self, items: List[tuple[str,str]]):
        # items: list of N (prompt, response) string pairs
        # sort key len(p[0]) = character count of the prompt -> shortest first
        self.items = sorted(items, key=lambda p: len(p[0]))
        self._i = 0                                  # cursor into self.items
    def __iter__(self):
        self._i = 0                                  # reset cursor -> re-iterable (one fresh pass)
        return self
    def __next__(self):
        if self._i >= len(self.items):               # walked past the last item?
            raise StopIteration
        it = self.items[self._i]                     # current (prompt, response) pair
        self._i += 1                                 # advance for the next call
        return it