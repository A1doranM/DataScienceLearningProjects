"""Part 9 entry point — run the GRPO unit test, then an optional tiny GRPO demo.

Repository layout (Part 9 -- RLHF with GRPO):

  part_9/
    orchestrator.py          # run the unit test + optional tiny GRPO demo  <-- THIS FILE
    policy.py                # policy = SFT LM + value head (head present but GRPO ignores it)
    rollout.py               # prompt formatting, sampling, logprobs/KL utilities
    grpo_loss.py             # policy-only clipped surrogate + KL penalty (NO value loss)
    train_grpo.py            # single-GPU RLHF loop with the GROUP-RELATIVE baseline
    eval_ppo.py              # compare reward vs. reference on a small set (Part-8 name, reused)
    tests/
      test_grpo_loss.py      # the policy-only objective returns a scalar

Run from inside `part_9/`:
    cd part_9
    python orchestrator.py          # runs the unit test
    python orchestrator.py --demo   # test, then a tiny GRPO run + eval (needs Part 6 + Part 7 ckpts)

Notes:
  - The actual commands below correctly target train_grpo.py and tests/test_grpo_loss.py.
  - The demo requires Part 6 SFT and Part 7 RM checkpoints; without them, run the test only.
"""

import argparse, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(cmd.split(), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny GRPO demo")
    args = p.parse_args()

    # 1) unit tests
    run("python -m pytest -q tests/test_grpo_loss.py")

    # 2) optional demo (requires SFT+RM checkpoints from Parts 6 & 7)
    if args.demo:
        run("python train_grpo.py --group_size 4 --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 200 --batch_prompts 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_ppo.py --policy_ckpt runs/grpo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 9 checks complete. ✅")
