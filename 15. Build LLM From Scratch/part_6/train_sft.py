"""6.5 SFT training loop — fine-tune a pretrained base model on (prompt, response) pairs.

What this file does
-------------------
This is the script that actually *runs* supervised fine-tuning (SFT). It
glues together every Part 6 piece you have built so far and then runs a
plain Part 4-style training loop on top of the Part 3 ``GPTModern`` model.

In plain English, ``main()`` does this:

    1. parse CLI args (steps, batch_size, model size, --ckpt, ...)
    2. load a tiny slice of SFT data       (dataset_sft.load_tiny_hf)
    3. sort it short -> long               (curriculum.LengthCurriculum)
    4. build the masking collator          (collator_sft.SFTCollator)
    5. build GPTModern (RMSNorm+SwiGLU+RoPE), optionally load --ckpt
    6. for `steps` steps:
           xb, yb = collate(batch)         # yb has -100 on prompt tokens
           logits, loss, _ = model(xb, yb) # masked-CE happens inside model
           loss.backward(); opt.step()
    7. save model + config to runs/sft/model_last.pt

The key idea of SFT vs pretraining is the *labels*, not the loss: the
collator already wrote ``-100`` over every prompt/question position, and
``F.cross_entropy`` (called inside ``GPTModern.forward``) skips those
positions automatically. So the model is only ever penalized for getting
the *answer* tokens wrong — it learns to respond, not to echo the prompt.

Loading ``--ckpt`` (a Part 4 base-model checkpoint) is what makes this
"fine-tuning" rather than "training from scratch": we start from weights
that already know language, then nudge them toward instruction-following.
If ``--ckpt`` is omitted the loop still runs, just from random init.

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
    [ 6.5 masked-CE training loop  (train_sft.py)    ]   <-- THIS FILE
              |
    [ 6.6 sample with the template (sample_sft.py)   ]
              |
    [ 6.7 evaluate EM / token-F1   (evaluate.py)     ]

Math
----
The loss is ordinary next-token cross-entropy, but averaged only over the
*unmasked* (answer) positions:

    loss = mean over { t : y_t != -100 } of  -log p(y_t | x_<=t)

where
    x_t   = input token id at position t            (B, T)
    y_t   = target token id at position t            (B, T)  (= x_{t+1}, shifted)
    -100  = the "ignore" label PyTorch's F.cross_entropy skips by default
    p(.)  = softmax(logits) from GPTModern

Because the collator set y_t = -100 for every prompt token, those terms
drop out of both the numerator (sum) and the count, so prompt tokens
contribute zero gradient. No special masking code is needed in this file.

Visualization
-------------
See notebook section 6.5 — the masked-CE training curve (loss printed
every 20 steps) and an illustration of which label positions are -100
(prompt, grey) vs scored (response, colored).

Shapes
------
  items                 : list[SFTItem] length 24      one tiny HF slice
  tuples                : list[(prompt, response)]      plain strings
  cur                   : list[(prompt, response)]      sorted short -> long
  batch                 : list[(prompt, response)] len <= batch_size
  xb (input ids)        : (B, block_size)               int64, pad id = 2
  yb (labels)           : (B, block_size)               int64, -100 = ignore
  logits                : (B, block_size, vocab_size)
  loss                  : scalar                        masked next-token CE
  saved checkpoint      : {'model': state_dict, 'config': cfg}

Key hyperparameters (argparse defaults)
---------------------------------------
  --steps      200      number of optimizer steps
  --batch_size 8        examples per step
  --block_size 256      max sequence length (must match the saved tokenizer)
  --n_layer    4        GPTModern depth
  --n_head     4        attention heads
  --n_embd     256      model width (d_model)
  --lr         3e-4     AdamW lr, betas=(0.9, 0.95), weight_decay=0.1
  --ckpt       None     optional Part 4 base checkpoint to fine-tune from
  --cpu        flag     force CPU even if CUDA is present
  --bpe_dir    ../part_4/.../tokenizer   trained BPE dir from Part 4
"""
from __future__ import annotations
import argparse, torch
import torch.nn as nn
from pathlib import Path
torch.manual_seed(0)

# Reuse GPTModern from Part 3
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_modern import GPTModern  # noqa: E402

from dataset_sft import load_tiny_hf
from collator_sft import SFTCollator
from curriculum import LengthCurriculum


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='huggingface', help='huggingface or path to local jsonl (unused in demo)')
    p.add_argument('--ckpt', type=str, required=False)
    p.add_argument('--out', type=str, default='runs/sft')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default='../part_4/runs/part4-demo/tokenizer') # assumes tokenizer exists from Part 4
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load a tiny HF slice or fallback examples
    items = load_tiny_hf(split='train[:24]', sample_dataset=False)  # list[SFTItem], len 24

    # Print few samples
    print(f"Loaded {len(items)} SFT items. Few samples:")
    for it in items[:3]:
        print(f"PROMPT: {it.prompt}\nRESPONSE: {it.response}\n{'-'*40}")

    # Curriculum over (prompt,response)
    tuples = [(it.prompt, it.response) for it in items]   # list[(str, str)]
    cur = list(LengthCurriculum(tuples))                  # same pairs, sorted short -> long
    print(cur)

    # Collator + model
    col = SFTCollator(block_size=args.block_size, bpe_dir=args.bpe_dir)
    model = GPTModern(vocab_size=col.vocab_size, block_size=args.block_size,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      use_rmsnorm=True, use_swiglu=True, rope=True).to(device)

    if args.ckpt:  # fine-tune from a Part 4 base checkpoint instead of random init
        print(f"Using model config from checkpoint {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = ckpt.get('config', {})
        model.load_state_dict(ckpt['model'])  # load pretrained weights into GPTModern

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()

    # Simple loop (single machine). We just cycle curriculum to fill batches, for a few steps.
    step = 0
    i = 0
    while step < args.steps:
        batch = cur[i:i+args.batch_size]
        if not batch:
            # restart curriculum
            # cur = list(LengthCurriculum(tuples)); 
            i = 0
            continue
        xb, yb = col.collate(batch)        # xb (B, block_size) ids, yb (B, block_size) labels (-100 = prompt)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss, _ = model(xb, yb)    # logits (B, block_size, vocab); loss = masked-CE scalar
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1; i += args.batch_size
        if step % 20 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = {
        "vocab_size": col.vocab_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": 0.0,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "rope": True,
        # tokenizer info (best-effort)
        "tokenizer_type": "byte" if col.vocab_size == 256 else "bpe",
        "tokenizer_dir": None,   # set a real path if you have a trained BPE dir
    }
    torch.save({'model': model.state_dict(), 'config': cfg},
               str(Path(args.out)/'model_last.pt'))
    print(f"Saved SFT checkpoint to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()