"""Byte-level BPE tokenizer (handout section 2.6): turns text into token IDs, and token IDs back into text."""

import re

from cs336_basics.bpe import Pair


class Tokenizer:
    """Encodes text into token IDs and decodes them back, with the vocabulary and merges that `train_bpe` learned."""

    def __init__(self, vocab: dict[int, bytes], merges: list[Pair], special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)  # ID -> bytes, as train_bpe gave it
        self.ids: dict[bytes, int] = {}  # bytes -> ID: the same table, reversed
        for token_id, token in self.vocab.items():
            self.ids[token] = token_id
        self.ranks: dict[Pair, int] = {}  # pair -> its place in merges, i.e. when train_bpe learned it
        for rank, pair in enumerate(merges):
            self.ranks[pair] = rank
        self.special_tokens = list(special_tokens or [])  # None becomes []; our own copy of the list
        for special in self.special_tokens:
            special_bytes = special.encode("utf-8")
            if special_bytes not in self.ids:
                new_id = max(self.vocab) + 1  # one past the biggest ID: always free
                self.vocab[new_id] = special_bytes
                self.ids[special_bytes] = new_id
        # The pattern that cuts a text at its special tokens and keeps them: the ( ) make re.split return each token
        # it cut at. Longest first, so "<|endoftext|><|endoftext|>" stays one token instead of two "<|endoftext|>".
        self.special_pattern: str | None = None  # stays None with no special tokens: "()" would cut everywhere
        if self.special_tokens:
            # A lambda rather than key=len: with key=len, ty types the sorted tokens as `Sized` instead of `str`.
            longest_first = sorted(self.special_tokens, key=lambda sp: len(sp), reverse=True)
            escaped: list[str] = []
            for token in longest_first:
                escaped.append(re.escape(token))  # "|" means "or" in a pattern; re.escape makes it a plain "|"
            self.special_pattern = "(" + "|".join(escaped) + ")"
