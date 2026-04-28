"""2.1 Byte-level tokenizer -- text becomes token IDs.

What this file does
-------------------
Maps Python strings to raw UTF-8 bytes and back. Each byte is an integer token
in the fixed vocabulary 0..255, so the vocabulary size is always 256 and no
training step is needed for the tokenizer.

Where this fits in the Part 2 training pipeline
-----------------------------------------------
    [ Raw text file / prompt string ]   <-- THIS FILE
                  |
    [ Byte token IDs: 0..255 ]          <-- THIS FILE
                  |
    [ 2.2 Dataset windows x/y ]
                  |
    [ 2.3 GPT forward pass ]
                  |
    [ 2.4 Cross-entropy + optimizer ]
                  |
    [ 2.5 Sampling / generation ]
                  |
    [ 2.6 Checkpoint + eval ]

Connection to Part 1 and later parts
------------------------------------
Part 1 started at embeddings: tensors shaped (B, T, d_model). Part 2 shows how
real text gets converted into the integer IDs that feed those embeddings.
Part 4 replaces this byte tokenizer with BPE, but the downstream model still
receives integer token IDs with shape (B, T).

Math / mapping
--------------
    text string --UTF-8 encode--> bytes b_0, b_1, ..., b_{N-1}
    token_id_i = integer value of byte b_i, where token_id_i is in [0, 255]
    decode(token_ids) = bytes(token_ids).decode("utf-8", errors="ignore")

Shapes
------
    encode(str) -> LongTensor [N]
    decode([N]) -> str
    vocab_size = 256

Visualization
-------------
See notebook section 2.1 -- Byte Tokenizer. It traces "I love deep learning"
from text to UTF-8 bytes to token IDs, then shows why non-English text becomes
multiple bytes per character.
"""

from __future__ import annotations
import torch

class ByteTokenizer:
    """Ultra-simple byte-level tokenizer.
    - encode(str) -> LongTensor [N]
    - decode(Tensor[int]) -> str
    - vocab_size = 256
    """
    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        return 256
