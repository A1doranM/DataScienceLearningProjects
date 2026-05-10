"""Byte-level tokenizer (vocab=256) — same as Part 2.

What this file does
-------------------
Encodes a Python string by taking its UTF-8 bytes; decodes by reversing.
Vocabulary is fixed at 256 (one ID per possible byte value). No training,
no merges — the simplest possible tokenizer.

This is a stand-in so Part 3 can be self-contained. In Part 4 it will be
replaced by a trained BPE tokenizer with a much larger vocab.

Shapes
------
  encode("abc") -> torch.LongTensor of shape (3,)   = [97, 98, 99]
  decode([97, 98, 99]) -> "abc"
"""
from __future__ import annotations
import torch

class ByteTokenizer:
    """Simple byte-level tokenizer (0..255)."""
    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)
    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(ids).decode('utf-8', errors='ignore')
    @property
    def vocab_size(self) -> int:
        return 256