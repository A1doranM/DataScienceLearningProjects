"""6.6 SampleSFT — generate an answer from a fine-tuned checkpoint using the chat template.

What this file does
-------------------
This is the "talk to your model" script. After Part 6's masked-CE training loop
has fine-tuned the Part 4 base model on (instruction, response) pairs, this file
loads that checkpoint and lets you type a single instruction and watch the model
answer. The trick that makes the answer come out clean is that we wrap your raw
instruction in the *same* chat template the model saw during training, but with an
empty response slot, so the model is primed to continue right where "### Response:"
begins.

    your text  ->  format_prompt_only(text)         # template, empty response
               ->  .replace('</s>','')              # drop the trailing end marker
               ->  col.encode(...)                  # text -> token ids
               ->  torch.tensor([ids])              # (1, T) batch of one
               ->  model.generate(...)              # autoregressive sampling
               ->  col.tok.decode(out_ids)          # ids -> readable text
               ->  print(...)

Note on the end marker: format_prompt_only renders "...### Response:</s>" (the
template always closes with "</s>"). We strip that trailing "</s>" so the model
is left *open* to generate the response instead of seeing an already-finished turn.
Early stopping is handled inside GPTModern.generate, whose eos_id defaults to 1:
once the model emits token id 1 it stops, so it can decide when the answer is done.

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
    [ 6.6 sample with the template (sample_sft.py)   ]   <-- THIS FILE
              |
    [ 6.7 evaluate EM / token-F1   (evaluate.py)     ]

Visualization
-------------
See notebook section 6.6 — shows a prompt going through the template, the
generated continuation, and the decoded answer text.

Shapes
------
  prompt_text : str                 templated instruction, trailing "</s>" removed
  ids         : list[int], len T    token ids from the SFT collator's tokenizer
  idx         : (1, T)              long tensor, batch of one, on `device`
  out         : (1, T + G)          generate() appends up to G = args.tokens ids
                                    (G may be smaller if eos_id=1 stops it early)
  out_ids     : list[int], len T+G  out[0].tolist()
  generated   : str                 decoded text that gets printed

Decode path
-----------
  if the tokenizer exposes .decode  -> decode the FULL out_ids (prompt + answer)
  else (raw byte fallback)          -> decode only the suffix out_ids[orig_len:]
                                       where orig_len = T (the prompt length)
"""
from __future__ import annotations
import argparse, torch

# Reuse GPTModern
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_modern import GPTModern  # noqa: E402

from collator_sft import SFTCollator
from formatters import format_prompt_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--tokens', type=int, default=80)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default='../part_4/runs/part4-demo/tokenizer')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)   # fine-tuned weights + saved config
    cfg = ckpt.get('config', {})

    col = SFTCollator(block_size=cfg.get('block_size', 256), bpe_dir=args.bpe_dir)  # gives us encode() + .tok
    model = GPTModern(vocab_size=col.vocab_size, block_size=args.block_size,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      use_rmsnorm=True, use_swiglu=True, rope=True).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    prompt_text = format_prompt_only(args.prompt).replace('</s>','')  # template, empty response, end marker dropped
    ids = col.encode(prompt_text)                       # str -> list[int], len T
    idx = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T) batch of one

    with torch.no_grad():
        # generate() appends up to args.tokens ids; its eos_id defaults to 1 (early stop)
        out = model.generate(idx, max_new_tokens=args.tokens,   # out: (1, T + G), G <= args.tokens
                             temperature=args.temperature, top_k=3)

    # decode: prefer BPE if collator has it, else fall back to bytes
    out_ids = out[0].tolist()       # (T + G,) ids: prompt followed by generated answer
    orig_len = idx.size(1)          # T = prompt length, marks where generation starts
    if hasattr(col, "tok") and hasattr(col.tok, "decode"):
        # decode full text or just the generated suffix; suffix is often clearer
        generated = col.tok.decode(out_ids)             # full prompt + answer back to text
        print(generated)
    else:
        generated = bytes(out_ids[orig_len:]).decode("utf-8", errors="ignore")  # byte fallback: answer only
        print(generated)


if __name__ == '__main__':
    main()