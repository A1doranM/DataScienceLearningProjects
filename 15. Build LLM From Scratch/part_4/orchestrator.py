"""Part 4 orchestrator — run unit tests, optionally a smoke train + sample.

Repository layout (Part 4)
--------------------------
    part_4/
      orchestrator.py             # this file: pytest + optional smoke train
      tokenizer_bpe.py            # 4.1 BPE tokenization (train/save/load)
      dataset_bpe.py              # 4.1 streaming dataset + (x,y) shift
      lr_scheduler.py             # 4.2 Warmup + cosine decay scheduler
      amp_accum.py                # 4.3 AMP + grad accumulation helpers
      checkpointing.py            # 4.4 save/resume + TB logging helpers
      logger.py                   # 4.5 logging backends (TB / W&B / Noop)
      train.py                    # 4.6 main training loop
      sample.py                   # 4.7 load checkpoint & generate text
      tests/
        test_tokenizer_bpe.py     # BPE train/save/load roundtrip
        test_scheduler.py         # warmup-cosine LR progression
        test_resume_shapes.py     # checkpoint save/load roundtrip

Walkthrough notebook: part_4_walkthrough.ipynb (sections 4.1 - 4.7).

Downstream dependency
---------------------
Parts 6 (SFT), 7 (Reward Modeling), 8 (PPO), 9 (GRPO) all expect a
trained Part 4 BPE tokenizer + base checkpoint at `runs/part4-demo/`.
Run `python orchestrator.py --demo` once before exploring those parts.

Run from inside part_4/:
    cd part_4
    python orchestrator.py            # tests only
    python orchestrator.py --demo     # tests + smoke train+sample
    pytest -q                         # tests directly
    tensorboard --logdir=runs/part4-demo
"""
import argparse, pathlib, subprocess, sys, shlex

ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run a tiny smoke train+sample")
    args = p.parse_args()

    # 1) unit tests
    run("python -m pytest -q tests/test_tokenizer_bpe.py")
    run("python -m pytest -q tests/test_scheduler.py")
    run("python -m pytest -q tests/test_resume_shapes.py")

    # 2) optional demo (quick overfit on tiny file)
    if args.demo:
        run("python train.py --data ../part_2/tiny.txt --out runs/part4-demo --bpe --vocab_size 8000 --epochs 1 --steps 300 --batch_size 16 --block_size 128 --n_layer 2 --n_head 2 --n_embd 128 --mixed_precision --grad_accum_steps 2 --log tensorboard")
        run("python sample.py --ckpt runs/part4-demo/model_last.pt --tokens 100 --prompt 'Generate a short story'")

    print("\nPart 4 checks complete. ✅")