"""Tests for cs336_basics/bpe.py, checked against the handout's own examples and rules (sections 2.4 and 2.5)."""

import pytest

from cs336_basics.bpe import learn_merges, merge_pair, split_on_special_tokens

EOT = "<|endoftext|>"  # the separator between documents in the course's TinyStories and OpenWebText files


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


# Section 2.5: special tokens are hard boundaries. The text is cut at each one, and the token itself is dropped.
# (tests/adapters.py still says special tokens in the input are "treated as any other string". That line is out of
# date: section 2.5 and test_train_bpe_special_tokens both drop them.)


def test_split_cuts_at_each_special_token_and_drops_it() -> None:
    # The course's TinyStories files put a newline on each side of the separator, and those newlines are text.
    text = f"the cat sat.\n{EOT}\nthe hat sat."
    assert split_on_special_tokens(text, [EOT]) == ["the cat sat.\n", "\nthe hat sat."]


def test_split_reads_special_tokens_literally_not_as_regex_patterns() -> None:
    # As a regex, [SEP] means "one of S, E or P", just as the | in <|endoftext|> means "or".
    assert split_on_special_tokens("cats[SEP]dogs", ["[SEP]"]) == ["cats", "dogs"]


def test_split_cuts_at_every_special_token_in_the_list() -> None:
    assert split_on_special_tokens(f"a{EOT}b<|pad|>c", [EOT, "<|pad|>"]) == ["a", "b", "c"]


def test_split_cuts_at_every_occurrence_however_many_there_are() -> None:
    # The course's files hold thousands of stories. A split that stops after a few cuts glues the rest together,
    # e.g. re.split(pattern, text, re.MULTILINE): the third argument of re.split is maxsplit, so that is maxsplit=8.
    stories = [f"story {i}." for i in range(20)]
    assert split_on_special_tokens(EOT.join(stories), [EOT]) == stories


def test_split_with_no_special_tokens_keeps_the_text_whole() -> None:
    # "|".join([]) is the empty pattern, and splitting on it would cut between every two characters.
    assert split_on_special_tokens("the cat sat.", []) == ["the cat sat."]


@pytest.mark.parametrize("special_tokens", [["<|x|>", "<|x|>!"], ["<|x|>!", "<|x|>"]])
def test_split_prefers_the_longer_special_token_whatever_the_list_order(special_tokens: list[str]) -> None:
    # This goes beyond the handout's "|".join recipe. Trying the tokens in list order leaves "!b" behind with the
    # first list, and trying the last-listed one first does it with the second. The "!" belongs to the special
    # token and must not reach the counts; section 2.6's test_overlapping_special_tokens needs the same rule.
    # The shorter token is still special on its own, so the bare "<|x|>" before "c" is cut too.
    assert split_on_special_tokens("a<|x|>!b<|x|>c", special_tokens) == ["a", "b", "c"]


def test_split_leaves_the_callers_list_of_special_tokens_unchanged() -> None:
    # Sort a copy (sorted), not the caller's list (.sort): the vocabulary numbers special tokens in the caller's order.
    special_tokens = ["<|x|>", "<|x|>!"]
    split_on_special_tokens("a<|x|>!b", special_tokens)
    assert special_tokens == ["<|x|>", "<|x|>!"]


def test_split_handles_special_tokens_at_the_ends_and_side_by_side() -> None:
    # Empty pieces ("") are fine, since they hold no pre-tokens, so only the non-empty ones are compared.
    pieces = split_on_special_tokens(f"{EOT}a{EOT}{EOT}b{EOT}", [EOT])
    assert [piece for piece in pieces if piece] == ["a", "b"]


def test_split_keeps_text_that_is_only_whitespace() -> None:
    # An empty TinyStories story leaves exactly this between two separators, and those newlines are still text.
    pieces = split_on_special_tokens(f"a{EOT}\n\n{EOT}b", [EOT])
    assert [piece for piece in pieces if piece] == ["a", "\n\n", "b"]
