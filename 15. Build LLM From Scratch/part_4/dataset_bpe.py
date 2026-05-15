"""4.1 Dataset — tokenize a text file once, slice into (x, y) shifted pairs.

What this file does
-------------------
Reads a `.txt` file end-to-end, runs the BPE tokenizer once over it, and
keeps the resulting long `LongTensor` of token IDs in memory. Then each
`__getitem__(i)` returns the standard causal LM training pair:

    x = ids[i     : i + block_size]      # (block_size,)
    y = ids[i + 1 : i + block_size + 1]  # (block_size,)

i.e. `y` is `x` shifted by one position — the next-token prediction
target every position needs to predict.

Where this fits in the training pipeline
----------------------------------------
    [ raw text file ]
              |
    [ 4.1 BPE Tokenizer            ]
              |
    [ 4.1 Dataset / DataLoader     ]   <-- THIS FILE
              |
    [ forward: GPTModern (Part 3)  ]
              ...

The (x, y) shift
----------------
For a sentence "The cat sat", BPE might give ids = [12, 88, 41, 7]. With
block_size=3 the dataset yields:

    x = [12, 88, 41]    "The cat sat"
    y = [88, 41,  7]    " cat sat <next>"

At inference, position t in `x` should predict `y[t]` -> the loss is
`F.cross_entropy(logits[t], y[t])`. The model itself is causal, so
position t only sees x[0..t].

DataLoader wraps this dataset with `batch_size`, shuffling, and
`drop_last=True` so every batch is full.

Shapes
------
  ids in memory               : (N,)        N = number of BPE tokens in file
  __getitem__(i) returns      : two (block_size,) tensors  (x, y)
  Dataloader collated batch   : (batch_size, block_size)   each of (x, y)
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple
from tokenizer_bpe import BPETokenizer

class TextBPEBuffer(Dataset):
    """Memory-mapped-ish single-file dataset: tokenize once → long tensor of ids.
    get(idx) returns a (block_size,) slice; we construct (x,y) with shift inside collate.
    """
    def __init__(self, path: str, tokenizer: BPETokenizer, block_size: int = 256):
        super().__init__()
        self.block_size = block_size
        text = Path(path).read_text(encoding='utf-8')
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    def __len__(self):
        return max(0, self.ids.numel() - self.block_size - 1)
    def __getitem__(self, i: int):
        x = self.ids[i:i+self.block_size]
        y = self.ids[i+1:i+self.block_size+1]
        return x, y

def make_loader(path: str, tokenizer: BPETokenizer, block_size: int, batch_size: int, shuffle=True) -> DataLoader:
    ds = TextBPEBuffer(path, tokenizer, block_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)