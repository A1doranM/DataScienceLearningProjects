"""Byte-level BPE training (handout sections 2.4-2.5): the merge loop that turns pre-token counts into merges."""

import json
import multiprocessing
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from itertools import pairwise
from typing import BinaryIO

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


# From the course's cs336_basics/pretokenization_example.py, unchanged. That file can't be imported: the usage
# example at its bottom runs on import.
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _count_chunk(job: tuple[str | os.PathLike, int, int, list[str]]) -> dict[Word, int]:
    """Count the pre-tokens in bytes start..end of the file. Runs inside one worker process."""
    input_path, start, end, special_tokens = job
    with open(input_path, "rb") as f:  # bytes, because start and end are byte positions
        f.seek(start)
        data = f.read(end - start)
    chunk = data.decode("utf-8")
    return count_pre_tokens(chunk, special_tokens)


def count_pre_tokens_in_file(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[Word, int]:
    """Count the pre-tokens of the file at `input_path`: the same counts as count_pre_tokens on its whole text.

    The file is cut just before a special token (the first one in the list), so no pre-token spans two chunks,
    and the chunks are counted in parallel worker processes. Without special tokens, the file is one chunk.

    The workers start from a clean "forkserver" process, not as copies ("fork") of the caller: a copy of a process
    that already runs other threads can deadlock. As with any start method but fork, a script that calls this must
    keep its top-level code under `if __name__ == "__main__":`.
    """
    num_processes = os.cpu_count() or 1
    if special_tokens:
        # More chunks than workers: each chunk stays small, and every worker stays busy until the end.
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, 4 * num_processes, special_tokens[0].encode("utf-8"))
    else:
        boundaries = [0, os.path.getsize(input_path)]
    jobs = [(input_path, start, end, special_tokens) for start, end in pairwise(boundaries)]
    if len(jobs) <= 1:
        # Nothing to split (e.g. no special token in the file): starting worker processes would cost more (about
        # 0.2 s) than counting the one chunk right here.
        results = [_count_chunk(job) for job in jobs]
    else:
        with multiprocessing.get_context("forkserver").Pool(min(num_processes, len(jobs))) as pool:
            results = pool.map(_count_chunk, jobs)
    total: Counter[Word] = Counter()
    for word_counts in results:
        total.update(word_counts)
    return dict(total)


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
    words = list(word_counts)
    counts = list(word_counts.values())
    # Count every pair once. `where` maps each pair to the positions of the words that contain it.
    pair_counts: Counter[Pair] = Counter()
    where: defaultdict[Pair, set[int]] = defaultdict(set)
    for i, word in enumerate(words):
        for pair in pairwise(word):
            pair_counts[pair] += counts[i]
            where[pair].add(i)
    merges: list[Pair] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        _, best = max((count, pair) for pair, count in pair_counts.items())  # highest count, then greater pair
        merges.append(best)
        # Update only the words that contain `best`: take out their old pairs, merge, put in their new pairs.
        # Loop over a copy, because `where[best]` shrinks inside the loop.
        for i in list(where[best]):
            word = words[i]
            count = counts[i]
            for pair in pairwise(word):
                pair_counts[pair] -= count
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                where[pair].discard(i)
            word = merge_pair(word, best)
            for pair in pairwise(word):
                pair_counts[pair] += counts[i]
                where[pair].add(i)
            words[i] = word
    return merges


def build_vocab(special_tokens: list[str], merges: list[Pair]) -> dict[int, bytes]:
    """Give every token an ID: the special tokens first, then the 256 single bytes, then each merge result.

    The order follows the handout's own example (section 2.4). Special tokens get 0, 1, ... in the caller's order,
    so their IDs do not depend on the vocabulary size. Byte b gets ID len(special_tokens) + b. Each merge's new token
    (its two halves joined) gets the next ID, in the order the merges were learned.
    """
    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    for first, second in merges:
        vocab[len(vocab)] = first + second
    return vocab


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a byte-level BPE tokenizer on the text file at `input_path`, and return its vocabulary and merges.

    `vocab_size` is the largest vocabulary allowed: the special tokens, plus the 256 bytes, plus one entry per
    merge. Training stops early, with fewer merges, if the text runs out of pairs. Special tokens are hard
    boundaries that are never counted (section 2.5), and each gets an ID of its own. Raises ValueError when
    `vocab_size` is too small to hold even the 256 bytes and the special tokens.

    The file is pre-tokenized in parallel worker processes (`count_pre_tokens_in_file`), so a script that calls
    this must keep its top-level code under `if __name__ == "__main__":`.
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}, got {vocab_size}")
    word_counts = count_pre_tokens_in_file(input_path, special_tokens)
    number_of_merges = vocab_size - len(special_tokens) - 256
    merges = learn_merges(word_counts, number_of_merges)
    vocab = build_vocab(special_tokens, merges)
    return (vocab, merges)


# From the course's tests/common.py, unchanged (the package can't import from tests/).
@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def _to_text(token: bytes) -> str:
    """One token as GPT-2 text: every byte becomes one visible character, e.g. b" the" -> "Ġthe"."""
    byte_encoder = gpt2_bytes_to_unicode()
    text = ""
    for b in token:
        text = text + byte_encoder[b]
    return text


def save_vocab_and_merges(
    vocab: dict[int, bytes], merges: list[Pair], vocab_path: str | os.PathLike, merges_path: str | os.PathLike
) -> None:
    """Write the vocabulary and merges in GPT-2's format, the format of the course's reference files.

    `vocab_path` gets one JSON object {token: id}; `merges_path` gets one merge per line, its two halves separated by
    a space. Tokens are written with `_to_text`, so every byte is one visible character (a space becomes "Ġ"), and
    the one real space on a merges line is always the separator.
    """
    vocab_json: dict[str, int] = {}
    for token_id, token in vocab.items():
        vocab_json[_to_text(token)] = token_id
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        for first, second in merges:
            first_text = _to_text(first)
            second_text = _to_text(second)
            f.write(first_text + " " + second_text + "\n")


def _to_bytes(text: str) -> bytes:
    """One token's GPT-2 text back to its bytes, e.g. "Ġthe" -> b" the". The reverse of `_to_text`."""
    byte_decoder: dict[str, int] = {}  # char -> byte: gpt2_bytes_to_unicode() reversed, e.g. "Ġ" -> 32
    for b, char in gpt2_bytes_to_unicode().items():  # (32, "Ġ"), (97, "a"), ...
        byte_decoder[char] = b  # "Ġ" -> 32
    values: list[int] = []
    for char in text:  # "Ġt": "Ġ", then "t"
        values.append(byte_decoder[char])  # "Ġ" -> 32, "t" -> 116
    return bytes(values)  # [32, 116] -> b" t"


def load_vocab_and_merges(
    vocab_path: str | os.PathLike, merges_path: str | os.PathLike
) -> tuple[dict[int, bytes], list[Pair]]:
    """Read back the files `save_vocab_and_merges` wrote (GPT-2's format): the reverse of saving."""
    with open(vocab_path, encoding="utf-8") as f:
        vocab_json: dict[str, int] = json.load(f)  # {"<|endoftext|>": 0, "Ā": 1, ..., "Ġt": 257, ...}
    vocab: dict[int, bytes] = {}
    for text, token_id in vocab_json.items():  # ("Ġt", 257)
        vocab[token_id] = _to_bytes(text)  # vocab[257] = b" t"
    merges: list[Pair] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:  # "Ġ t\n", then "h e\n", ...
            if line.startswith("#version"):  # GPT-2's own merges.txt starts with this header line; ours doesn't
                continue
            first_text, second_text = line.rstrip("\n").split(" ")  # "Ġ t\n" -> "Ġ", "t"
            first = _to_bytes(first_text)  # "Ġ" -> b" "
            second = _to_bytes(second_text)  # "t" -> b"t"
            merges.append((first, second))  # (b" ", b"t")
    return vocab, merges
