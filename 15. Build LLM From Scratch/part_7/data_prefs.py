"""7.1 PreferenceData — load (prompt, chosen, rejected) pairs for reward modeling.

What this file does
-------------------
Reward modeling needs *comparisons*, not labels: for the same situation a human
says "answer A is better than answer B." This file produces those comparisons as
``PrefExample`` records, each holding three strings::

    PrefExample(prompt, chosen, rejected)
                  |        |        |
              the setup  the good  the worse
                         answer    answer

``load_preferences`` first tries to download a small slice of the real
Anthropic/hh-rlhf dataset. In that dataset each row is a *full conversation*
already stitched together under the keys ``chosen`` and ``rejected`` (there is no
separate prompt field), so we store the whole conversation in those two fields and
leave ``prompt`` empty. We also skip any row where either side is empty after
stripping. If the ``datasets`` library is missing, or the download fails, or the
slice yields nothing usable, we fall back to 2 hand-written toy pairs so the rest
of Part 7 can still run end to end offline.

Where this fits in the Part 7 reward-modeling pipeline
------------------------------------------------------
    preference pair (prompt, chosen, rejected)        (data_prefs.py)   <-- THIS FILE
              |
    [ plate both with SFT template + tokenize         (collator_rm.py)  ]
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
See notebook section 7.1 — how a preference pair (prompt, chosen, rejected) is
loaded and what the chosen-vs-rejected text looks like for a few examples.

Shapes
------
  load_preferences(split) : returns List[PrefExample], length up to slice size
  PrefExample.prompt      : str  (empty "" for HH rows; a real prompt for toy pairs)
  PrefExample.chosen      : str  (the preferred / better response or conversation)
  PrefExample.rejected    : str  (the dispreferred / worse response or conversation)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

@dataclass
class PrefExample:
    prompt: str    # the setup/context; "" for HH rows (conversation lives in chosen/rejected)
    chosen: str    # the preferred ("better") response or full conversation
    rejected: str  # the dispreferred ("worse") response or full conversation


def load_preferences(split: str = "train[:200]") -> List[PrefExample]:
    """Load a tiny preference set. Tries Anthropic HH; falls back to a toy set.
    HH fields: 'chosen', 'rejected' (full conversations). We use an empty prompt.
    """
    items: List[PrefExample] = []
    if load_dataset is not None:  # only attempt download if the datasets lib imported
        try:
            ds = load_dataset("Anthropic/hh-rlhf", split=split)  # e.g. "train[:200]"
            for row in ds:
                # HH stores full conversations under 'chosen'/'rejected'; no prompt field
                ch = str(row.get("chosen", "")).strip()
                rj = str(row.get("rejected", "")).strip()
                if ch and rj:  # empty-side filter: skip rows missing either response
                    items.append(PrefExample(prompt="", chosen=ch, rejected=rj))
        except Exception:
            print("Failed to load Anthropic/hh-rlhf dataset. Using fallback toy pairs.")
            pass
    if not items:
        # fallback toy pairs: 2 hand-written examples so Part 7 runs fully offline
        items = [
            PrefExample("Summarize: Scaling laws for neural language models.",
                        "Scaling laws describe how performance improves predictably as model size, data, and compute increase.",
                        "Scaling laws are when you scale pictures to look bigger."),
            PrefExample("Give two uses of attention in transformers.",
                        "It lets the model focus on relevant tokens and enables parallel context integration across positions.",
                        "It remembers all past words exactly without any computation."),
        ]
    return items