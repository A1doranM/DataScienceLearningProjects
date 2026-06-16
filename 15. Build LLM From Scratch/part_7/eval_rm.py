"""7.4 EvalRewardModel — score a trained reward checkpoint with pairwise accuracy.

What this file does
-------------------
Loads a trained reward-model checkpoint and measures how often it ranks the
*chosen* answer above the *rejected* one on held-out preference pairs. For each
pair we tokenize both sides, push them through the model to get two scalar
rewards, and check whether the chosen one scored higher. The fraction of pairs
where that is true is the "pairwise accuracy" — a single number that tells you
how well the reward model learned human preferences (0.5 = random guessing,
1.0 = perfect). In pseudo-code:

    for each preference pair (prompt, chosen, rejected):
        r_pos = model(tokenize(prompt, chosen))     # scalar reward for chosen
        r_neg = model(tokenize(prompt, rejected))   # scalar reward for rejected
        correct += (r_pos > r_neg)
    accuracy = correct / total

No gradients, no training — this is pure inference under torch.no_grad(), with
the model in .eval() mode (so dropout is off).

Where this fits in the Part 7 reward-modeling pipeline
------------------------------------------------------
    preference pair (prompt, chosen, rejected)        (data_prefs.py)
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
    [ eval: pairwise accuracy r_pos > r_neg           (eval_rm.py)      ]   <-- THIS FILE

Math
----
  accuracy = mean( 1[ r_pos > r_neg ] )

    r_pos     : scalar reward the model gives the chosen answer
    r_neg     : scalar reward the model gives the rejected answer
    1[ . ]    : indicator, 1 when the condition holds else 0
    mean(.)   : average over all evaluated preference pairs

  This is exactly the win rate of "chosen beats rejected": no loss, no log,
  just count the wins and divide by the total number of pairs.

Visualization
-------------
See notebook section 7.4 — visualizes the reward gap (r_pos - r_neg) across
pairs and how the fraction with a positive gap (the accuracy) settles as the
model trains.

Shapes
------
  triples            : list of (prompt, chosen, rejected) strings    (one per pair)
  batch              : up to B=16 triples per loop step
  pos, neg           : (B, block_size=256)  long token ids, pad id 2 (from collator)
  r_pos, r_neg       : (B,)                  one scalar reward per sequence
  (r_pos > r_neg)    : (B,)                  bool per pair; summed into `correct`
  acc                : scalar               correct / total over all pairs
"""
from __future__ import annotations
import argparse, torch
from data_prefs import load_preferences
from collator_rm import PairCollator
from model_reward import RewardModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--split', type=str, default='val[:200]')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    items = load_preferences(split=args.split)
    triples = [(it.prompt, it.chosen, it.rejected) for it in items]  # list of (prompt, chosen, rejected)

    col = PairCollator(block_size=256, bpe_dir=args.bpe_dir)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})

    model = RewardModel(vocab_size=cfg.get('vocab_size', col.vocab_size), block_size=cfg.get('block_size', 256),
                        n_layer=cfg.get('n_layer', 4), n_head=cfg.get('n_head', 4), n_embd=cfg.get('n_embd', 256))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    # Evaluate accuracy r_pos>r_neg
    import math
    B = 16
    correct = 0; total = 0
    for i in range(0, len(triples), B):
        batch = triples[i:i+B]                       # up to B triples
        pos, neg = col.collate(batch)                # each (B, block_size=256) padded token ids
        pos, neg = pos.to(device), neg.to(device)
        with torch.no_grad():
            r_pos = model(pos)                       # (B,) reward for chosen
            r_neg = model(neg)                       # (B,) reward for rejected
        correct += (r_pos > r_neg).sum().item()      # count pairs where chosen wins
        total += pos.size(0)
    acc = correct / max(1, total)
    print(f"pairs={total}  accuracy (r_pos>r_neg) = {acc:.3f}")

if __name__ == '__main__':
    main()