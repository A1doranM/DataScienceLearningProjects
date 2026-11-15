"""Tests for cs336_basics/bpe.py: the BPE merge loop, checked against the handout's own examples (section 2.4)."""

import pytest

from cs336_basics.bpe import learn_merges, merge_pair


def as_word(text: str) -> tuple[bytes, ...]:
    """'low' -> (b'l', b'o', b'w'): a pre-token before any merge, one token per UTF-8 byte."""
    return tuple(bytes([byte]) for byte in text.encode("utf-8"))


# The handout's toy corpus (p. 7) after splitting on whitespace, and the merges the handout lists for it.
TOY_WORD_COUNTS = {as_word("low"): 5, as_word("lower"): 2, as_word("widest"): 3, as_word("newest"): 6}
TOY_MERGES = [
    tuple(merge.encode().split())
    for merge in ["s t", "e st", "o w", "l ow", "w est", "n e", "ne west", "w i", "wi d", "wid est", "low e", "lowe r"]
]


def test_toy_corpus_gives_the_handouts_merges_in_order() -> None:
    assert learn_merges(TOY_WORD_COUNTS, num_merges=12) == TOY_MERGES


def test_six_toy_merges_split_newest_into_ne_west() -> None:
    word = as_word("newest")
    for pair in learn_merges(TOY_WORD_COUNTS, num_merges=6):
        word = merge_pair(word, pair)
    assert word == (b"ne", b"west")


def test_training_stops_when_every_pre_token_is_one_token() -> None:
    # After the 12th merge each toy word is a single token, so no pair is left for a 13th.
    assert learn_merges(TOY_WORD_COUNTS, num_merges=100) == TOY_MERGES


@pytest.mark.parametrize(
    ("word_counts", "expected"),
    [
        # The handout's example (p. 6): these four pairs are tied, and BPE merges ("BA", "A").
        ({(b"A", b"B"): 1, (b"A", b"C"): 1, (b"B", b"ZZ"): 1, (b"BA", b"A"): 1}, (b"BA", b"A")),
        # Same first token, so the second one decides, in whichever order the pre-tokens come.
        ({as_word("ab"): 1, as_word("ac"): 1}, (b"a", b"c")),
        ({as_word("ac"): 1, as_word("ab"): 1}, (b"a", b"c")),
    ],
)
def test_a_tie_goes_to_the_lexicographically_greater_pair(
    word_counts: dict[tuple[bytes, ...], int], expected: tuple[bytes, bytes]
) -> None:
    assert learn_merges(word_counts, num_merges=1) == [expected]


def test_pair_counts_are_weighted_by_how_often_the_pre_token_occurs() -> None:
    # (y, z) would win a tie, but (a, b) is inside a pre-token that occurs twice.
    assert learn_merges({as_word("yz"): 1, as_word("ab"): 2}, num_merges=1) == [(b"a", b"b")]


def test_every_occurrence_in_a_pre_token_counts_overlapping_ones_too() -> None:
    # "aaa" holds (a, a) twice, so it beats (b, c), which would win a tie.
    assert learn_merges({as_word("aaa"): 1, as_word("bc"): 1}, num_merges=1) == [(b"a", b"a")]


@pytest.mark.parametrize(("word_counts", "num_merges"), [({}, 5), (TOY_WORD_COUNTS, 0)])
def test_no_pre_tokens_or_zero_merges_gives_no_merges(
    word_counts: dict[tuple[bytes, ...], int], num_merges: int
) -> None:
    assert learn_merges(word_counts, num_merges) == []


@pytest.mark.parametrize(
    ("text", "pair", "expected"),
    [
        ("low", (b"o", b"w"), (b"l", b"ow")),
        ("abab", (b"a", b"b"), (b"ab", b"ab")),  # every occurrence is merged
        ("aaa", (b"a", b"a"), (b"aa", b"a")),  # overlapping occurrences: the leftmost wins
        ("low", (b"w", b"o"), (b"l", b"o", b"w")),  # pair absent: the word is unchanged
    ],
)
def test_merge_pair_joins_each_occurrence_left_to_right(
    text: str, pair: tuple[bytes, bytes], expected: tuple[bytes, ...]
) -> None:
    assert merge_pair(as_word(text), pair) == expected
