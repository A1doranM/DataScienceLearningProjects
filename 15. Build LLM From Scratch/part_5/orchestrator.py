"""Part 5 orchestrator — run unit tests, optionally a tiny MoE demo.

Repository layout (Part 5)
--------------------------
    part_5/
      orchestrator.py            # this file: pytest + optional demo
      README.md                  # compact concept notes (theory / distributed / hybrid)
      gating.py                  # 5.1 router (top-k) + 5.2 load-balancing aux loss
      experts.py                 # 5.3 expert MLPs (SwiGLU or GELU)
      moe.py                     # 5.4 MoE layer (dispatch/combine)
      block_hybrid.py            # 5.5 hybrid dense+MoE FFN
      demo_moe.py                # 5.6 forward demo + routing histogram
      tests/
        test_gate_shapes.py      # gate idx/weights shapes, 0 <= w <= 1
        test_moe_forward.py      # shape preservation + gradient flow
        test_hybrid_block.py     # hybrid blend shape, aux >= 0

Walkthrough notebook: part_5_walkthrough.ipynb (sections 5.1 - 5.6).

Scope note
----------
Part 5 is an isolated *component*, not a full model: no attention, no
training loop. The MoE layer is a drop-in replacement for the dense FFN
sublayer of any Transformer block (see notebook section 5.6 for the
integration with Part 3's modern block).

Run from inside part_5/:
    cd part_5
    python orchestrator.py            # tests only
    python orchestrator.py --demo     # tests + MoE routing demo
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
    p.add_argument("--demo", action="store_true", help="run a tiny MoE demo")
    args = p.parse_args()

    # 1) unit tests
    run("python -m pytest -q tests/test_gate_shapes.py")
    run("python -m pytest -q tests/test_moe_forward.py")
    run("python -m pytest -q tests/test_hybrid_block.py")

    # 2) optional demo
    if args.demo:
        run("python demo_moe.py --tokens 6 --hidden 128 --experts 4 --top_k 1")

    print("\nPart 5 checks complete. ✅")