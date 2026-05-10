"""3.8 demo_generate.py — time cached vs nocache generation.

What this file does
-------------------
Builds an *un-trained* GPTModern with the modernization flags you choose,
then runs two generations from the same prompt:

  1) `model.generate(...)`         — with KV cache + sliding window + sink
  2) `model.generate_nocache(...)` — recomputes the full window each step

It prints elapsed time for each, plus the decoded output. Because the
model is untrained, the text is gibberish — the point is the *speed gap*
and that the two paths produce identical token sequences (when sampling
is deterministic via temperature=0).

Where this fits in the modern Transformer block
-----------------------------------------------
Exercises the full GPTModern (3.7) in inference mode:

    [ Token IDs (B, 1) ]
            |
    [ GPTModern.forward(idx, kv_cache_list, start_pos) ]   <-- THIS FILE
            |
    [ next-token logits -> top-k/p sample -> append -> repeat ]

How to run
----------
    cd part_3
    python demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200

Flags
-----
  --rmsnorm          : use RMSNorm instead of LayerNorm
  --rope             : use RoPE inside attention
  --swiglu           : use SwiGLU FFN instead of GELU MLP
  --sliding_window N : crop K/V to last N positions (per layer)
  --sink S           : keep first S tokens always (attention sink)
  --group_size G     : n_kv_head = n_head / G   (GQA)
  --tokens N         : how many new tokens to generate
  --cpu              : force CPU even if CUDA is available

Expected behavior
-----------------
With cache, generation should be roughly N x faster than nocache for
N new tokens, since past K/V are reused instead of recomputed.
"""
import argparse, torch
from tokenizer import ByteTokenizer
from model_modern import GPTModern
import time

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--rmsnorm', action='store_true')
    p.add_argument('--rope', action='store_true')
    p.add_argument('--swiglu', action='store_true')
    p.add_argument('--sliding_window', type=int, default=None)
    p.add_argument('--sink', type=int, default=0)
    p.add_argument('--group_size', type=int, default=2)
    p.add_argument('--tokens', type=int, default=120)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    tok = ByteTokenizer()
    model = GPTModern(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=4, n_embd=128,
                      use_rmsnorm=args.rmsnorm, use_swiglu=args.swiglu, rope=args.rope,
                      max_pos=4096, sliding_window=args.sliding_window, attention_sink=args.sink, n_kv_head=args.group_size).to(device)

    # empty prompt → newline
    prompt = torch.tensor([[10]], dtype=torch.long, device=device)

    with torch.no_grad():
        start = time.time()
        out = model.generate(prompt, max_new_tokens=args.tokens, temperature=0.0, top_k=50, top_p=None,
                              sliding_window=args.sliding_window, attention_sink=args.sink)
        print(f"Generated {args.tokens} tokens in {time.time()-start:.2f} sec")

        start = time.time()
        out_nocache = model.generate_nocache(prompt, max_new_tokens=args.tokens, temperature=0.0, top_k=50, top_p=None,
                              sliding_window=args.sliding_window, attention_sink=args.sink)
        print(f"(nocache) Generated {args.tokens} tokens in {time.time()-start:.2f} sec")
    print(tok.decode(out[0].cpu()))
    print(tok.decode(out_nocache[0].cpu()))