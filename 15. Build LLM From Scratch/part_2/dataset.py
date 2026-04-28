"""2.2 Byte dataset -- training windows and next-token labels.

What this file does
-------------------
Loads a text file as raw bytes, splits it into train/val buffers, and samples
random language-model batches. Each batch returns input IDs x and target IDs y
where y is exactly x shifted one byte to the right.

Where this fits in the Part 2 training pipeline
-----------------------------------------------
    [ Raw text bytes ]
          |
    [ train / val split ]
          |
    [ sample block windows ]           <-- THIS FILE
          |
    [ x: current tokens, y: next tokens ] <-- THIS FILE
          |
    [ GPT(x) -> logits ]
          |
    [ cross_entropy(logits, y) ]

Connection to Part 1 and later parts
------------------------------------
Part 1 explained how a Transformer block processes a sequence tensor. This
file creates the sequence windows that become that tensor after embedding.
Part 4 replaces this with a BPE dataset, but the same x/y shift is preserved.
Parts 6-9 keep the same causal-LM idea, with extra masking or reward signals.

Math
----
For a sampled start index s and block_size T:

    x[t] = data[s + t]          for t = 0..T-1
    y[t] = data[s + t + 1]      for t = 0..T-1

The model is trained to predict y[t], the next token after x[t].

Shapes
------
    train, val:     LongTensor [N]
    x:              LongTensor [B, T]
    y:              LongTensor [B, T]

Visualization
-------------
See notebook section 2.2 -- Dataset + Label Shift. It aligns x and y in a
table so the one-token shift is visible cell by cell.
"""

from __future__ import annotations
from pathlib import Path
import torch

class ByteDataset:
    """Holds raw bytes of a text file and yields (x,y) blocks for LM.
    - block_size: sequence length (context window)
    - split: fraction for training (rest is val)
    """
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        data = Path(path).read_bytes()
        data = torch.tensor(list(data), dtype=torch.long)
        n = int(len(data) * split)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(self, which: str, batch_size: int, device: torch.device):
        buf = self.train if which == 'train' else self.val
        assert len(buf) > self.block_size + 1, 'file too small for given block_size'
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
        # x is the current context window: bytes at positions [i, i+T).
        x = torch.stack([buf[i:i+self.block_size] for i in ix])
        # y is the next-token target window: the same slice shifted by +1.
        y = torch.stack([buf[i+1:i+1+self.block_size] for i in ix])
        return x.to(device), y.to(device)
