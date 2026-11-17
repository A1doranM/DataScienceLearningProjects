"""Byte-level BPE training (handout sections 2.4-2.5): the merge loop that turns pre-token counts into merges."""

import re
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise

import regex

type Word = tuple[bytes, ...]  # one pre-token as its current tokens: (b"l", b"o", b"w"), later (b"l", b"ow")
type Pair = tuple[bytes, bytes]  # two adjacent tokens, e.g. (b"o", b"w")

# GPT-2's pre-tokenizer (handout section 2.4). Use it with the `regex` module: the built-in `re` has no \p{...}.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Cut `text` at every special token and drop the tokens, keeping all the text between them, in order.

    Each piece is pre-tokenized on its own, so no pre-token, and so no merge, ever spans a special token, and the
    special tokens themselves never reach the counts. With no special tokens the whole text is one piece. When one
    special token starts with another ("<|x|>" and "<|x|>!"), the longer one wins, whatever the list order. Text
    that is only whitespace is still text and is kept. A special token at either end, or two in a row, may leave an
    empty piece: harmless, since an empty piece holds no pre-tokens. The caller's `special_tokens` list is not changed.
    """
    if not special_tokens:
        return [text]
    # A lambda rather than key=len: with key=len, ty types the sorted tokens as `Sized` instead of `str`.
    longest_first = sorted(special_tokens, key=lambda token: len(token), reverse=True)
    pattern = "|".join(re.escape(token) for token in longest_first)
    return re.split(pattern, text)


def count_pre_tokens(text: str, special_tokens: list[str]) -> dict[Word, int]:
    """Count the pre-tokens of `text`, each split into single bytes: the `word_counts` that `learn_merges` takes.

    The text is first cut at the special tokens, so no pre-token spans one. Each piece is then pre-tokenized with
    GPT-2's regex `PAT`, and every pre-token is counted. Each distinct pre-token becomes one key: its UTF-8 bytes as
    a tuple of single bytes, e.g. " the" -> (b" ", b"t", b"h", b"e").
    """
    counts: Counter[str] = Counter()
    for piece in split_on_special_tokens(text, special_tokens):
        for match in regex.finditer(PAT, piece):
            pre_token = match.group()
            counts[pre_token] += 1
    word_counts: dict[Word, int] = {}
    for pre_token, count in counts.items():
        word = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
        word_counts[word] = count
    return word_counts


def merge_pair(word: Word, pair: Pair) -> Word:
    """Join every occurrence of `pair` in `word` into one token, scanning left to right."""
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if word[i : i + 2] == pair:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def learn_merges(word_counts: Mapping[Word, int], num_merges: int) -> list[Pair]:
    """Learn up to `num_merges` BPE merges from pre-token counts, in the order they are made.

    `word_counts` maps each pre-token, split into its tokens (single bytes before training), to how often it
    occurs. Each pre-token appears once, so no two keys spell the same bytes.

    Each round counts every adjacent pair inside every pre-token, weighted by that pre-token's count.
    Overlapping pairs count too ("aaa" holds (a, a) twice), but a pair never spans two pre-tokens. The most
    frequent pair is merged everywhere, and a tie goes to the lexicographically greater pair. Training stops
    early once every pre-token is a single token, so fewer than `num_merges` merges can come back.
    """
    merges: list[Pair] = []
    for _ in range(num_merges):
        pair_counts: Counter[Pair] = Counter()
        for word, count in word_counts.items():
            for pair in pairwise(word):
                pair_counts[pair] += count
        if not pair_counts:
            break
        _, best = max((count, pair) for pair, count in pair_counts.items())  # highest count, then greater pair
        merges.append(best)
        # Merging only regroups a pre-token's bytes, so two different pre-tokens never become the same key.
        word_counts = {merge_pair(word, best): count for word, count in word_counts.items()}
    return merges
