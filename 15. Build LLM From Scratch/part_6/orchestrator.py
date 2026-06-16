"""6.0 Orchestrator — the Part 6 entry point: run the tests, then optionally train + sample.

What this file does
-------------------
This is the "press play" script for Part 6 (Supervised Fine-Tuning / SFT).
It does two things, in order:

  1) Always: run the unit tests under ``tests/`` (the chat-template formatter
     test and the label-masking test) by shelling out to ``pytest``.
  2) Only with ``--demo``: run a tiny end-to-end SFT demo by shelling out to
     ``train_sft.py`` (fine-tune the Part 4 base checkpoint for a few hundred
     steps) and then ``sample_sft.py`` (generate answers for a few example
     instructions using the chat template).

It does NOT define any model or do any training itself. The helper ``run(cmd)``
just prints the command, launches it as a subprocess with cwd set to this
folder (``part_6/``), and aborts the whole run if any sub-command fails:

    run("...")  ->  print ">>> cmd"  ->  subprocess.run(...)  ->  exit if returncode != 0

So the whole file is a thin "glue" layer that wires together the other Part 6
modules (formatters, dataset, collator, curriculum, train, sample) into one
reproducible command.

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
    [ 6.5 masked-CE training loop  (train_sft.py)    ]
              |
    [ 6.6 sample with the template (sample_sft.py)   ]
              |
    [ 6.7 evaluate EM / token-F1   (evaluate.py)     ]

    ^^^ THIS FILE drives the ENTIRE pipeline above   <-- THIS FILE
        (runs the tests, then with --demo calls train_sft.py + sample_sft.py)

Visualization
-------------
See the whole Part 6 notebook — it walks through every box above (chat
template, masking, curriculum, the SFT loop, sampling, and EM/F1 evaluation),
of which this script is the runnable "do it all" entry point.

How to run
----------
Run from inside ``part_6/`` so the relative paths (``tests/``, ``train_sft.py``,
``sample_sft.py``, and ``../part_4/...``) resolve correctly:

    cd part_6
    python orchestrator.py            # just the unit tests
    python orchestrator.py --demo     # tests + tiny train + a few sample generations
    pytest -q                         # (optional) run the tests directly

The ``--demo`` path assumes the Part 4 demo has already been run, because it
fine-tunes from ``../part_4/runs/part4-demo/model_last.pt``.
"""

# Repository layout (Part 6)
#
#   part_6/
#     orchestrator.py           # run unit tests + optional tiny SFT demo   <-- THIS FILE
#     formatters.py             # 6.1 prompt/response templates
#     dataset_sft.py            # HF dataset loader (+tiny fallback) → (prompt, response)
#     collator_sft.py           # 6.2 causal LM labels with masking
#     curriculum.py             # 6.3 length‑based curriculum sampler
#     evaluate.py               # 6.4 simple exact/F1 metrics
#     train_sft.py              # minimal one‑GPU SFT loop (few steps)
#     sample_sft.py             # load ckpt & generate from instructions
#     tests/
#       test_formatter.py
#       test_masking.py

### FILE: part_6/orchestrator.py
import argparse, pathlib, subprocess, sys, shlex
ROOT = pathlib.Path(__file__).resolve().parent  # absolute path to part_6/ (so subprocesses run here)

def run(cmd: str):
    # Run one shell command as a subprocess, with cwd pinned to part_6/.
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)  # shlex.split -> argv list; cwd=ROOT keeps relative paths valid
    if res.returncode != 0:
        sys.exit(res.returncode)  # fail fast: propagate the child's exit code and stop the run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny SFT demo on a few samples")
    args = p.parse_args()

    # 1) unit tests  (always): chat-template formatting, then label masking
    run("python -m pytest -q tests/test_formatter.py")
    run("python -m pytest -q tests/test_masking.py")

    # 2) optional demo: tiny SFT train, then a few sample generations
    if args.demo:
        # --ckpt ../part_4/runs/part4-demo/model_last.pt # assumes Part 4 demo has been run
        run("python train_sft.py --data huggingface --ckpt ../part_4/runs/part4-demo/model_last.pt --out runs/sft-demo --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128")
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What are the three primary colors?' --tokens 30 --temperature 0.2")
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What does DNA stand for?' --tokens 30 --temperature 0.2")
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'Reverse engineer this code to create a new version\ndef factorialize(num):\n  factorial = 1\n  for i in range(1, num):\n    factorial *= i\n  \n  return factorial' --tokens 64 --temperature 0.2")

    print("\nPart 6 checks complete. ✅")