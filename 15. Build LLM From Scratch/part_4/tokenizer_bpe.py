"""4.1 BPE tokenizer — replace Part 2/3's byte tokenizer with a real subword one.

What this file does
-------------------
Wraps HuggingFace `ByteLevelBPETokenizer` so we can:
  * train BPE merges on one or more `.txt` files
  * save vocab.json + merges.txt + tokenizer.json + bpe_meta.json
  * reload later for both training and sampling
  * encode(text) -> List[int], decode(ids) -> str

Where this fits in the training pipeline
----------------------------------------
    [ raw text file ]
              |
    [ 4.1 BPE Tokenizer            ]   <-- THIS FILE
              |
    [ 4.1 Dataset / DataLoader     ]
              |
    [ forward: GPTModern (Part 3)  ]
              |
    [ CE loss                      ]
              |
    [ 4.3 AMP + grad accum         ]
              |
    [ AdamW.step()                 ]
              |
    [ 4.2 Warmup + Cosine LR       ]
              |
    [ 4.5 Logger / 4.4 Checkpoint  ]
              |
    [ next batch                   ]

Why BPE (vs byte tokenizer from Part 2/3)?
------------------------------------------
A byte tokenizer assigns one ID per UTF-8 byte (vocab=256). Every word
becomes ~4–8 tokens, so even a tiny corpus blows past the model's
`block_size`. BPE finds the most frequent byte pairs and merges them
into new tokens — common words and morphemes become a single ID.

Concretely, on the same English text:
  * byte tokenizer  : "learning"  -> 8 tokens
  * BPE (8 k vocab) : "learning"  -> 1 or 2 tokens

So BPE buys you ~4–8x effective context and ~4–8x cheaper training per
character. Below ~8k vocab the table is too small to capture much; above
~50k the embedding table dominates parameter count.

Saved artifacts
---------------
After `tok.save(dir)`, you'll find:
    dir/vocab.json       - {token: id}
    dir/merges.txt       - learned merge rules
    dir/tokenizer.json   - full HF Tokenizer state (preferred for load)
    dir/bpe_meta.json    - {vocab_size, special_tokens}

Shapes
------
  encode("I love deep learning") -> List[int] of length ~4 (typical for 8k+ vocab)
  decode([id, id, id, id])       -> "I love deep learning"
"""
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
except Exception:
    ByteLevelBPETokenizer = None

class BPETokenizer:
    """Minimal BPE wrapper (HuggingFace tokenizers).
    Trains on a text file or a folder of .txt files. Saves merges/vocab to out_dir.
    """
    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] | None = None):
        if ByteLevelBPETokenizer is None:
            raise ImportError("Please `pip install tokenizers` for BPETokenizer.")
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self, data_path: Union[str, Path]):
        files: List[str] = []
        p = Path(data_path)
        if p.is_dir():
            files = [str(fp) for fp in p.glob("**/*.txt")]
        else:
            files = [str(p)]
        tok = ByteLevelBPETokenizer()
        tok.train(files=files, vocab_size=self.vocab_size, min_frequency=2, special_tokens=self.special_tokens)
        self._tok = tok

    def save(self, out_dir: Union[str, Path]):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        assert self._tok is not None, "Train or load before save()."
        self._tok.save_model(str(out))
        self._tok.save(str(out / "tokenizer.json"))
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        (out/"bpe_meta.json").write_text(json.dumps(meta))

    def load(self, dir_path: Union[str, Path]):
        dirp = Path(dir_path)
        # Prefer explicit filenames; fall back to glob if needed.
        vocab = dirp / "vocab.json"
        merges = dirp / "merges.txt"
        tokenizer = dirp / "tokenizer.json"
        if not vocab.exists() or not merges.exists():
            # Fallback for custom basenames
            vs = list(dirp.glob("*.json"))
            ms = list(dirp.glob("*.txt"))
            if not vs or not ms:
                raise FileNotFoundError(f"Could not find vocab.json/merges.txt in {dirp}")
            vocab = vs[0]
            merges = ms[0]
        # tok = ByteLevelBPETokenizer(str(vocab), str(merges))
        tok = Tokenizer.from_file(str(tokenizer))
        self._tok = tok
        meta_file = dirp / "bpe_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.special_tokens = meta.get("special_tokens", self.special_tokens)


    def encode(self, text: str):
        ids = self._tok.encode(text).ids
        return ids

    def decode(self, ids):
        return self._tok.decode(ids)