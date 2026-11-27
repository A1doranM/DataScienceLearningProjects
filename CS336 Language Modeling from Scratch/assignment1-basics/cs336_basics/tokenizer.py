"""Byte-level BPE tokenizer (handout section 2.6): turns text into token IDs, and token IDs back into text."""

import os
import re
from collections.abc import Iterable, Iterator
from itertools import pairwise
from typing import Self

import regex

from cs336_basics.bpe import PAT, Pair, Word, load_vocab_and_merges, merge_pair

NEVER = float("inf")  # the rank of a pair that training never learned: higher than every real rank
CACHE_LIMIT = 1_000_000  # pieces encode remembers; past this, new pieces are still encoded, just not remembered


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
        # piece -> its IDs. Text repeats itself (TinyStories: 536,592,168 pieces, only 59,933 different), so each
        # different piece goes through the merges once, and every later time is one dict lookup.
        self._cache: dict[str, list[int]] = {}
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

    def decode(self, ids: list[int]) -> str:
        """Turn token IDs back into text. Bytes that aren't valid UTF-8 (e.g. half a character) become U+FFFD."""
        parts: list[bytes] = []
        for token_id in ids:
            parts.append(self.vocab[token_id])
        # Glue all the bytes first, then decode: a character can be split across tokens ("é" = b"\xc3" + b"\xa9").
        all_bytes = b"".join(parts)
        return all_bytes.decode("utf-8", errors="replace")

    def _encode_pre_token(self, pre_token: str) -> list[int]:
        """Encode one pre-token: split it into single bytes, then merge pairs in the order training learned them.

        The result is remembered in self._cache, and the same list is returned next time: callers must not change it.
        """
        try:
            return self._cache[pre_token]  # most pieces were seen before: one lookup
        except KeyError:
            pass
        word: Word = tuple(bytes([b]) for b in pre_token.encode("utf-8"))  # "the" -> (b"t", b"h", b"e")
        while True:
            # Find the pair in word that training learned first: the one with the lowest rank.
            best_pair: Pair | None = None
            best_rank = NEVER
            for pair in pairwise(word):
                rank = self.ranks.get(pair, NEVER)
                if rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:  # no pair in word was ever learned: nothing left to merge
                break
            word = merge_pair(word, best_pair)
        ids: list[int] = []
        for token in word:  # "the" ends as (b"the",)
            ids.append(self.ids[token])
        if len(self._cache) < CACHE_LIMIT:
            self._cache[pre_token] = ids
        return ids

    def _pieces(self, text: str) -> list[str]:
        """Cut text into encode's pieces: special tokens kept whole, the rest cut into pre-tokens by PAT."""
        # "the cat sat.<|endoftext|>the hat sat." -> ['the', ' cat', ' sat', '.', '<|endoftext|>', 'the', ...]
        if self.special_pattern is None:  # no special tokens: the whole text is one part
            parts = [text]
        else:
            parts = re.split(self.special_pattern, text)  # the ( ) in the pattern keep the special tokens
        pieces: list[str] = []
        for part in parts:  # 'the cat sat.', then '<|endoftext|>', then 'the hat sat.'
            if part in self.special_tokens:
                pieces.append(part)
            else:
                for match in regex.finditer(PAT, part):  # in 'the cat sat.': 'the', ' cat', ' sat', '.'
                    pieces.append(match.group())
        return pieces

    def _encode_pieces(self, pieces: list[str]) -> list[int]:
        """Turn pieces into IDs: a special token becomes its one ID, any other piece goes through the merges."""
        ids: list[int] = []
        for piece in pieces:  # 'the', ' cat', ' sat', '.', '<|endoftext|>', 'the', ...
            if piece in self.special_tokens:
                piece_bytes = piece.encode("utf-8")  # self.ids has bytes keys
                ids.append(self.ids[piece_bytes])
            else:
                ids.extend(self._encode_pre_token(piece))  # extend, not append: ' cat' adds 33, 100, 257
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs (handout section 2.6.1)."""
        pieces = self._pieces(text)
        return self._encode_pieces(pieces)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode a stream of text (e.g. a file, line by line) lazily, handing out one ID at a time.

        The last piece of each chunk is held back and put in front of the next chunk, because it may continue there
        (a word cut in two). So the IDs are the same as encode() of the whole text, but memory stays small. One chunk
        must not cut a special token in two; the lines of a file never do.
        """
        # chunks ["the ca", "t sat."] -> 259 | 33, 100, 257, 33, 116, 257 | 47  (= encode("the cat sat."))
        buffer = ""  # the held-back piece; empty at the start
        for chunk in iterable:  # "the ca", then "t sat."
            pieces = self._pieces(buffer + chunk)  # the held-back piece goes first: " ca" + "t sat."
            if pieces:
                buffer = pieces.pop()  # hold back the last piece: it may continue in the next chunk
            for token_id in self._encode_pieces(pieces):  # the IDs of the other pieces: ['the'] -> 259
                yield token_id
        last_pieces = self._pieces(buffer)  # the stream ended: what is left in buffer is complete now
        for token_id in self._encode_pieces(last_pieces):
            yield token_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Self:
        """Build a Tokenizer from the files `save_vocab_and_merges` wrote, e.g. our TinyStories vocab and merges."""
        vocab, merges = load_vocab_and_merges(vocab_filepath, merges_filepath)  # the two files -> vocab, merges
        return cls(vocab, merges, special_tokens)  # cls is the class it was called on, so this runs __init__
