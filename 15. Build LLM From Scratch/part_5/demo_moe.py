"""5.6 demo_moe.py — forward a random batch through the MoE, print routing stats.

What this file does
-------------------
Builds an *untrained* MoE layer with your chosen size, runs one forward
pass on random tokens, and prints:
  * the output shape (must equal the input shape — drop-in FFN property)
  * the load-balancing aux loss value
  * a primary-expert load histogram (how many tokens picked each expert
    as their top-1 choice)

With random init the router is arbitrary but usually *roughly* uniform —
you should see counts spread across experts, not all piled on one.

Where this fits
---------------
Exercises the full 5.1 -> 5.4 pipeline:

    [ random x (2, tokens/2, hidden) ]
              |
    [ MoE(dim, n_expert, k)          ]   <-- THIS FILE drives it
              |
    [ y + aux + routing histogram    ]

How to run
----------
    cd part_5
    python demo_moe.py --tokens 64 --hidden 128 --experts 4 --top_k 1

Flags
-----
  --tokens N    : total tokens (split into batch of 2)
  --hidden C    : model dim
  --experts E   : number of experts
  --top_k K     : experts per token
  --cpu         : force CPU
"""
import argparse, torch
from moe import MoE

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--experts', type=int, default=4)
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    x = torch.randn(2, args.tokens//2, args.hidden, device=device)  # (B=2,T=tokens/2,C)

    moe = MoE(dim=args.hidden, n_expert=args.experts, k=args.top_k).to(device)
    with torch.no_grad():
        y, aux = moe(x)

    # simple routing histogram
    from gating import TopKGate
    gate = moe.gate
    idx, w, _ = gate(x.view(-1, args.hidden))
    hist = torch.bincount(idx[:,0], minlength=args.experts)
    print(f"Output shape: {tuple(y.shape)} | aux={float(aux):.4f}")
    print("Primary expert load (counts):", hist.tolist())