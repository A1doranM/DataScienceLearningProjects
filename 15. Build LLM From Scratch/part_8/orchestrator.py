"""8.0 Orchestrator — the one button that runs all of Part 8 (tests, then the tiny PPO demo).

What this file does
-------------------
This is the *entry point* for Part 8. Think of it as the "run everything" button.
It does two jobs, in order:

  1. Always: run the unit tests in tests/ (proves the PPO loss math and the
     policy forward pass are correct before we trust the full loop).
  2. Only with --demo: run the real RLHF-PPO pipeline end to end —
     first train_ppo.py (fine-tune the policy with PPO), then eval_ppo.py
     (compare the tuned policy's average reward against the frozen reference).

Each step is just a shell command launched with subprocess.run; if any command
returns a non-zero exit code we stop immediately (sys.exit) so a failure is loud,
not silent. In tiny pseudo-code:

    run("pytest tests/test_ppo_loss.py")        # gate 1: loss math
    run("pytest tests/test_policy_forward.py")  # gate 2: policy shapes
    if --demo:
        run("python train_ppo.py ... --steps 100 ...")  # learn
        run("python eval_ppo.py  ...")                  # measure

The --demo path needs checkpoints produced earlier in the curriculum:
a Part 6 SFT model (the starting policy) and a Part 7 reward model (the judge),
plus the Part 4 BPE tokenizer. Without those files the demo cannot run.

The repository layout for Part 8 (so you know which file does what):

    part_8/
      orchestrator.py          # run unit tests + optional tiny PPO demo  <-- THIS FILE
      policy.py                # policy = SFT LM + value head (toy head on logits)
      rollout.py               # prompt formatting, sampling, logprobs/KL utilities
      ppo_loss.py              # PPO clipped objective + value + entropy + KL penalty
      train_ppo.py             # single-GPU RLHF loop (tiny, on-policy)
      eval_ppo.py              # compare reward vs. reference on a small set
      tests/
        test_ppo_loss.py
        test_policy_forward.py

Where this fits in the Part 8 RLHF-PPO loop
-------------------------------------------
    prompt
       |
    [ policy does a take: generate + record old_logp   (policy.py / rollout.py) ]
       |
    [ judge scores the take: reward model r            (part_7 RewardModel)     ]
       |
    [ KL leash: shaped_r = r - kl * (logp - ref_logp)  (vs frozen SFT reference)]
       |
    [ advantage = shaped_r - value baseline, normalized (train_ppo.py)          ]
       |
    [ PPO clipped update: min(unclipped, clipped)      (ppo_loss.py)            ]
       |
    [ AdamW step on the policy only                    (train_ppo.py)           ]
       |
    [ eval: avg reward, tuned policy vs frozen ref     (eval_ppo.py)            ]

    This file does not live in any single box above — it RUNS THE WHOLE LOOP:
    it launches the unit tests, then train_ppo.py (which drives every stage
    of the loop) and finally eval_ppo.py.   <-- THIS FILE

Visualization
-------------
See the whole Part 8 notebook — it walks through every stage that this
orchestrator launches (rollouts, reward + KL shaping, advantages, the clipped
PPO update, and the final eval), so running this file reproduces the notebook end to end.

How to run
----------
  Run from inside the part_8/ folder:

    cd part_8
    python orchestrator.py          # just the unit tests (no checkpoints needed)
    python orchestrator.py --demo   # tests + tiny PPO train/eval demo
    pytest -q                       # run the tests directly, without this script

  The --demo run requires the Part 6 SFT checkpoint and the Part 7 reward-model
  checkpoint (and the Part 4 BPE tokenizer) to already exist on disk.
"""

import argparse, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent  # part_8/ — all commands run with this as cwd

def run(cmd: str):
    # Launch one shell command and abort the whole script if it fails (non-zero exit).
    print(f"\n>>> {cmd}")
    res = subprocess.run(cmd.split(), cwd=ROOT)  # cmd.split(): "python x.py --a 1" -> ["python","x.py","--a","1"]
    if res.returncode != 0:
        sys.exit(res.returncode)  # propagate the failure code so CI / the shell sees it

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny PPO demo")
    args = p.parse_args()

    # 1) unit tests — always run; these gate the demo (must pass before we trust the loop)
    run("python -m pytest -q tests/test_ppo_loss.py")        # checks the clipped-surrogate math
    run("python -m pytest -q tests/test_policy_forward.py")  # checks policy logits/value shapes

    # 2) optional demo (requires SFT+RM checkpoints from Parts 6 & 7)
    if args.demo:
        # run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 10 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        # run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 50 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        # train: PPO-fine-tune the Part 6 SFT policy using the Part 7 reward model as judge
        run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 100 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # eval: score the just-trained policy's avg reward vs the frozen reference
        run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 8 checks complete. ✅")