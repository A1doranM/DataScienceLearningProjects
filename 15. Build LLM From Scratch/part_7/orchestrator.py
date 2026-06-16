"""7.0 Orchestrator — the single entry point that runs (and optionally demos) all of Part 7.

What this file does
-------------------
This is the "press play" script for the whole Part 7 reward-modeling lesson.
Run on its own it just executes the unit tests; pass ``--demo`` and it also
trains a tiny reward model and evaluates it on a few held-out pairs. Each step
is just a normal shell command that this file shells out to (via subprocess),
so you can also copy any printed ">>> ..." line and run it by hand.

    python orchestrator.py            # 1) run the two pytest files, then stop
    python orchestrator.py --demo     # 1) run tests, then 2) train + eval a tiny RM

Internally it is a thin wrapper:

    run(cmd) = print(cmd); subprocess.run(cmd, cwd=part_7/); exit if it failed

So the file owns no model, no tensors, no math of its own — it only sequences
the other Part 7 files (data_prefs -> collator_rm -> model_reward -> loss_reward
-> train_rm -> eval_rm) in the right order.

Repository layout (Part 7)

    part_7/
      orchestrator.py           # run unit tests + optional tiny RM demo  (THIS FILE)
      data_prefs.py             # 7.1 HF preference loader (+tiny fallback)
      collator_rm.py            # pairwise tokenization -> (pos, neg) tensors
      model_reward.py           # 7.2 reward model (Transformer encoder -> scalar)
      loss_reward.py            # 7.3 Bradley-Terry & margin-ranking losses
      train_rm.py               # minimal one-GPU training on tiny slice
      eval_rm.py                # 7.4 sanity checks & simple accuracy on val
      tests/
        test_bt_loss.py
        test_reward_forward.py

Where this fits in the Part 7 reward-modeling pipeline
------------------------------------------------------
    preference pair (prompt, chosen, rejected)        (data_prefs.py)
              |
    [ plate both with SFT template + tokenize         (collator_rm.py)  ]   <-- THIS FILE
    [   -> (pos_ids, neg_ids)                                           ]   <-- runs the
              |                                                             whole pipeline,
    [ reward model: encoder -> mean-pool -> scalar    (model_reward.py) ]   <-- not just one
    [   r_pos = score(chosen),  r_neg = score(rejected)                 ]   <-- box
              |
    [ Bradley-Terry loss on the gap                   (loss_reward.py)  ]   <-- THIS FILE
    [   softplus(-(r_pos - r_neg))                                      ]
              |
    [ train -> reward checkpoint                      (train_rm.py)     ]   <-- THIS FILE
              |
    [ eval: pairwise accuracy r_pos > r_neg           (eval_rm.py)      ]   <-- THIS FILE

How to run
----------
Run everything from inside the ``part_7/`` directory so the relative paths
(``tests/``, ``train_rm.py``, ``../part_4/...``) resolve correctly:

    cd part_7
    python orchestrator.py            # just the unit tests (fast)
    python orchestrator.py --demo     # tests + a tiny train (300 steps) + eval
    pytest -q                         # run the tests directly, without this wrapper

The ``--demo`` path expects a Part 4 BPE tokenizer at
``../part_4/runs/part4-demo/tokenizer`` and writes its checkpoint to
``runs/rm-demo/model_last.pt``. If any sub-command exits non-zero, ``run()``
propagates that exit code and the orchestrator stops immediately.

Visualization
-------------
See the Part 7 notebook — it walks the same reward-modeling pipeline end to end
(preference pairs -> pairwise tokenization -> reward model -> Bradley-Terry loss
-> train -> pairwise-accuracy eval) that this orchestrator wires together.
"""

import argparse, pathlib, subprocess, sys, shlex
ROOT = pathlib.Path(__file__).resolve().parent  # part_7/ — used as cwd for every sub-command

def run(cmd: str):
    print(f"\n>>> {cmd}")                          # echo the command so the demo is copy-pasteable
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)  # shlex.split -> argv list; run from part_7/
    if res.returncode != 0:
        sys.exit(res.returncode)                   # fail fast: stop the whole run on first error

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny reward‑model demo")
    args = p.parse_args()

    # 1) unit tests — always run (fast correctness checks)
    run("python -m pytest -q tests/test_bt_loss.py")        # Bradley-Terry loss behaves correctly
    run("python -m pytest -q tests/test_reward_forward.py") # reward model forward pass / shapes

    # 2) optional demo: tiny train + eval (only with --demo)
    if args.demo:
        run("python train_rm.py --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --loss bt --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split train[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split test[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 7 checks complete. ✅")