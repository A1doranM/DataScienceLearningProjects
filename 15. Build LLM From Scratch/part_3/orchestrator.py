"""Part 3 orchestrator — run all tests, optionally a tiny generation demo.

Repository layout (Part 3)
--------------------------
    part_3/
      orchestrator.py              # this file: pytest + optional demo
      tokenizer.py                 # local byte-level tokenizer (self-contained)
      rmsnorm.py                   # 3.1 RMSNorm
      rope_custom.py               # 3.2 RoPE cache + apply
      swiglu.py                    # 3.3 SwiGLU FFN
      kv_cache.py                  # 3.4 KV cache + rolling buffer
      attn_modern.py               # 3.5 attention w/ RoPE, GQA, sliding window, sink, KV cache
      block_modern.py              # 3.6 block = (RMSNorm|LN) + modern attention + (SwiGLU|GELU)
      model_modern.py              # 3.7 GPTModern wrapper with feature flags
      demo_generate.py             # 3.8 timing demo (cached vs nocache)
      tests/
        test_rmsnorm.py            # RMSNorm shape sanity
        test_rope_apply.py         # RoPE rotation (vanilla + GQA shapes)
        test_kvcache_shapes.py     # RollingKV bounded length

Walkthrough notebook: part_3_walkthrough.ipynb (sections 3.1 - 3.7).

Run from inside part_3/:
    cd part_3
    python orchestrator.py            # tests only
    python orchestrator.py --demo     # tests + generation demo
    pytest -q                         # tests directly
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
    p.add_argument("--demo", action="store_true", help="run a tiny generation demo")
    args = p.parse_args()

    # 1) run unit tests
    run("python -m pytest -q tests/test_rmsnorm.py")
    run("python -m pytest -q tests/test_rope_apply.py")
    run("python -m pytest -q tests/test_kvcache_shapes.py")

    # 2) (optional) generation demo
    if args.demo:
        run("python demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200")

    print("\nPart 3 checks complete. ✅")