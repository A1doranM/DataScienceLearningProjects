"""7.3 Reward-model training loop — fit a scalar reward from preference pairs.

What this file does
-------------------
This is the script that actually *runs* reward-model (RM) training. It glues
together every Part 7 piece you have built so far and runs a tiny training loop
whose only job is to make the model score the *chosen* answer higher than the
*rejected* one.

In plain English, ``main()`` does this:

    1. parse CLI args (steps, batch_size, model size, --loss, ...)
    2. load a tiny slice of preference data   (data_prefs.load_preferences)
       -> turn each item into a (prompt, chosen, rejected) triple
    3. build the pair collator                (collator_rm.PairCollator)
    4. build a FRESH RewardModel              (model_reward.RewardModel)
       (no checkpoint to load — the RM is trained from random init here)
    5. make an AdamW optimizer
    6. for `steps` steps:
           pos, neg = col.collate(batch)      # plate chosen / rejected with the SFT template
           r_pos = model(pos)                 # scalar reward for the chosen answer
           r_neg = model(neg)                 # scalar reward for the rejected answer
           loss  = bradley_terry_loss(r_pos, r_neg)   # or margin_ranking_loss
           loss.backward(); opt.step()
       and every 25 steps it prints loss + pairwise accuracy (r_pos > r_neg)
    7. save model + config to runs/.../model_last.pt

The whole point is the *gap* r_pos - r_neg, not the absolute reward value: the
Bradley-Terry loss is small only when chosen beats rejected by a wide margin, so
training pushes that gap positive. The logged accuracy ``(r_pos > r_neg)`` is the
fraction of the batch where the model already prefers the chosen answer.

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
    [ train -> reward checkpoint                      (train_rm.py)     ]   <-- THIS FILE
              |
    [ eval: pairwise accuracy r_pos > r_neg           (eval_rm.py)      ]

Math
----
Two interchangeable training objectives (pick with --loss); both reward a
*positive* gap d = r_pos - r_neg:

    Bradley-Terry (--loss bt, default):
        loss = mean of  -log sigma(r_pos - r_neg)
             = mean of  softplus(-(r_pos - r_neg))

    Margin ranking (--loss margin):
        loss = mean of  max(0, -(r_pos - r_neg) + margin)      (margin = 1.0)

where
    r_pos     = scalar reward for the chosen answer            (B,)
    r_neg     = scalar reward for the rejected answer          (B,)
    sigma(.)  = logistic sigmoid
    softplus(z) = log(1 + e^z)  (the numerically-stable -log sigma form)
    margin    = how much higher r_pos must be before the term hits zero

Pairwise accuracy logged during training is just the fraction of the batch with
r_pos > r_neg.

Visualization
-------------
See notebook section 7.3 — the reward-model training curve (loss + pairwise
accuracy printed every 25 steps) and how the chosen/rejected reward gap widens
as training proceeds.

Shapes
------
  triples      : list of (prompt, chosen, rejected) strings   (80 items here)
  batch        : list of B such triples                       (B = batch_size = 8)
  pos / neg    : (B, T) long token ids   (T = block_size = 256, pad id = 2)
  r_pos / r_neg: (B,)    scalar reward per sequence
  loss         : ()      scalar (mean over the batch)
"""
from __future__ import annotations
import argparse, torch
from pathlib import Path

from data_prefs import load_preferences
from collator_rm import PairCollator
from model_reward import RewardModel
from loss_reward import bradley_terry_loss, margin_ranking_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/rm-demo')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--loss', choices=['bt','margin'], default='bt')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # data
    items = load_preferences(split='train[:80]')                  # tiny preference set (<= 80 PrefExample)
    triples = [(it.prompt, it.chosen, it.rejected) for it in items]  # list of (prompt, chosen, rejected) strings

    # collator + model
    col = PairCollator(block_size=args.block_size, bpe_dir=args.bpe_dir)
    model = RewardModel(vocab_size=col.vocab_size, block_size=args.block_size,  # FRESH RM: trained from random init (no --ckpt)
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # train (tiny)
    step = 0; i = 0
    while step < args.steps:
        batch = triples[i:i+args.batch_size]                      # up to B triples
        if not batch:
            i = 0; continue                                       # wrap around to reuse the tiny dataset
        pos, neg = col.collate(batch)                             # pos, neg: (B, T) long token ids (pad id = 2)
        pos, neg = pos.to(device), neg.to(device)
        r_pos = model(pos)                                        # (B,) reward for the chosen answer
        r_neg = model(neg)                                        # (B,) reward for the rejected answer
        if args.loss == 'bt':
            loss = bradley_terry_loss(r_pos, r_neg)               # () scalar: softplus(-(r_pos - r_neg)).mean()
        else:
            loss = margin_ranking_loss(r_pos, r_neg, margin=1.0)  # () scalar: hinge on the gap
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1; i += args.batch_size
        if step % 25 == 0:
            acc = (r_pos > r_neg).float().mean().item()           # pairwise accuracy: fraction with r_pos > r_neg
            print(f"step {step}: loss={loss.item():.4f} acc={acc:.2f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': {
        'vocab_size': col.vocab_size,
        'block_size': args.block_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved reward model to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()