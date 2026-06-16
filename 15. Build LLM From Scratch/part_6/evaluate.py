"""6.7 evaluate — score generated answers with exact-match and token-F1.

What this file does
-------------------
After fine-tuning (Part 6) you let the model generate an answer string and you
want a NUMBER that says "how close is this to the reference answer?". This file
gives two tiny, classic text metrics that take two plain strings (the model's
``pred`` and the human ``gold``) and return a float:

  _normalize("Hello, World!")  -> "hello world"   # lowercase + punct->space
  exact_match(pred, gold)      -> 1.0 if the normalized strings are identical
  token_f1(pred, gold)         -> overlap of words, balancing precision & recall

Both metrics first run ``_normalize`` so that differences in case, punctuation,
or extra whitespace do not count against the model. There is no neural network
here at all -- just string cleanup and counting shared words.

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
    [ 6.4 curriculum: easy -> hard (curriculum.py)   ]
              |
    [ 6.5 masked-CE training loop  (train_sft.py)    ]
              |
    [ 6.6 sample with the template (sample_sft.py)   ]
              |
    [ 6.7 evaluate EM / token-F1   (evaluate.py)     ]   <-- THIS FILE

Math
----
Let P = the multiset (bag) of normalized words in ``pred``
    G = the multiset (bag) of normalized words in ``gold``
    common = number of words shared between P and G, counted WITH multiplicity
             (each ``gold`` word can be matched at most once -- see the gp.remove
             loop below, which deletes a matched word so duplicates are not
             double-counted)

  exact_match = 1.0  if normalize(pred) == normalize(gold)  else 0.0

  precision   = common / |P|          (of the words the model said, how many were right)
  recall      = common / |G|          (of the words it should have said, how many it got)
  token_f1    = 2 * precision * recall / (precision + recall)   (harmonic mean)

  Edge cases handled in code:
    both empty            -> F1 = 1.0   (nothing to say, said nothing: perfect)
    exactly one empty     -> F1 = 0.0   (no possible overlap)
    common == 0           -> F1 = 0.0   (avoids 0/0 in the formula)

Note on _normalize: it lowercases and replaces every character that is NOT a
letter, digit, or whitespace with a SPACE (not with nothing). Consequences:
  - an apostrophe SPLITS a word into two tokens: "don't" -> "don" + "t"
  - "C++" COLLAPSES to "c"  (the two '+' become spaces, then trimmed away)
This is the standard SQuAD-style normalization and is intentionally simple.

Visualization
-------------
See notebook section 6.7 -- walks through normalizing a sample pred/gold pair and
shows how precision, recall, and the F1 harmonic mean fall out of the shared-word
count, including the apostrophe-split and "C++"->"c" gotchas.

Shapes
------
  pred, gold : str                       (raw model output / reference answer)
  _normalize : str  -> str               (cleaned single-line lowercase string)
  .split()   : str  -> List[str]         (p, g: lists of word tokens)
  exact_match: (str, str) -> float       (scalar, 0.0 or 1.0)
  token_f1   : (str, str) -> float       (scalar in [0.0, 1.0])
"""
from __future__ import annotations
import re
from typing import List, Tuple

def _normalize(s: str) -> str:
    s = s.lower()                          # case-fold so "Cat" == "cat"
    s = re.sub(r"[^a-z0-9\s]", " ", s)     # punctuation -> SPACE (apostrophe splits, "C++"->"c ")
    s = re.sub(r"\s+", " ", s).strip()     # collapse runs of whitespace to one, trim ends
    return s

def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))   # 1.0 if identical strings, else 0.0

def token_f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()           # List[str]: predicted word tokens (bag P)
    g = _normalize(gold).split()           # List[str]: gold word tokens (bag G)
    if not p and not g:
        return 1.0                         # both empty -> perfect match
    if not p or not g:
        return 0.0                         # exactly one empty -> no overlap possible
    common = 0
    gp = g.copy()                          # mutable copy of gold; consume matches once each
    for t in p:
        if t in gp:
            gp.remove(t); common += 1      # remove() prevents double-counting duplicate words
    if common == 0:
        return 0.0                         # no shared words -> guard against 0/0
    prec = common / len(p)                 # precision: right words / predicted words
    rec  = common / len(g)                 # recall:    right words / gold words
    return 2*prec*rec/(prec+rec)           # F1 = harmonic mean of precision and recall