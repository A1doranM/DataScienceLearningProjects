"""2.6 Evaluate validation loss -- measure a saved checkpoint.

What this file does
-------------------
Loads a checkpoint, reconstructs the GPT from its saved config, samples many
validation batches, and prints the average cross-entropy loss.

Where this fits in the Part 2 training pipeline
-----------------------------------------------
    [ validation bytes ]
              |
    [ saved checkpoint ]
              |
    [ repeated val batches ]       <-- THIS FILE
              |
    [ average cross-entropy ]      <-- THIS FILE

Connection to Part 1 and later parts
------------------------------------
Part 1 had correctness tests for individual math pieces. Part 2 starts using
loss as a model-quality signal. Part 4 adds more complete checkpointing and
logging; Parts 6-9 add task-specific evaluation and reward metrics.

Math
----
    val_loss = mean_i CE(model(x_i), y_i)

where each (x_i, y_i) is a sampled validation batch.

Shapes
------
    x, y:    LongTensor [B, T]
    logits:  FloatTensor [B, T, vocab_size]
    loss:    scalar

Visualization
-------------
See notebook section 2.6 -- Checkpoints + Eval. It shows how model_best.pt
becomes the artifact future parts load or replace.
"""

from __future__ import annotations
import argparse, torch
from dataset import ByteDataset
from model_gpt import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ds = ByteDataset(args.data, block_size=args.block_size)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {
        'vocab_size': 256,
        'block_size': args.block_size,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 256,
        'dropout': 0.0,
    })
    model = GPT(**cfg).to(device)
    model.load_state_dict(ckpt['model'])

    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.iters):
            xb, yb = ds.get_batch('val', args.batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    print(f"val loss: {sum(losses)/len(losses):.4f}")


if __name__ == '__main__':
    main()
