"""Tests for cs336_basics.tokenizer_experiments: a whole file encoded to uint16 token IDs, and the compression ratio."""

import numpy as np
import pytest

from cs336_basics.bpe import save_vocab_and_merges
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.tokenizer_experiments import compression_ratio, encode_file, sample_documents

from .common import FIXTURES_PATH

EOT = "<|endoftext|>"


def gpt2_tokenizer() -> Tokenizer:
    return Tokenizer.from_files(FIXTURES_PATH / "gpt2_vocab.json", FIXTURES_PATH / "gpt2_merges.txt", [EOT])


def test_encode_file_in_many_chunks_matches_encode(tmp_path):
    tokenizer = gpt2_tokenizer()
    source = FIXTURES_PATH / "tinystories_sample.txt"  # 3,794 bytes, 5 stories
    out = tmp_path / "ids.bin"
    count = encode_file(tokenizer, source, out, chunk_bytes=512)  # small chunks: several workers, cut at <|endoftext|>
    expected = tokenizer.encode(source.read_text(encoding="utf-8"))
    ids = np.fromfile(out, dtype=np.uint16)
    assert count == len(expected)
    assert out.stat().st_size == 2 * count  # 2 bytes per ID
    assert ids.tolist() == expected


def test_encode_file_without_special_token_is_one_chunk(tmp_path):
    tokenizer = gpt2_tokenizer()
    source = tmp_path / "plain.txt"
    source.write_text("the cat sat. the hat sat.\n", encoding="utf-8")
    out = tmp_path / "ids.bin"
    count = encode_file(tokenizer, source, out, chunk_bytes=4)
    assert np.fromfile(out, dtype=np.uint16).tolist() == tokenizer.encode("the cat sat. the hat sat.\n")
    assert count == len(tokenizer.encode("the cat sat. the hat sat.\n"))


def test_encode_file_refuses_ids_that_do_not_fit_in_uint16(tmp_path):
    vocab = {i: i.to_bytes(3, "big") for i in range(2**16 + 1)}  # IDs 0 .. 65,536: the last one needs 17 bits
    save_vocab_and_merges(vocab, [], tmp_path / "vocab.json", tmp_path / "merges.txt")
    tokenizer = Tokenizer.from_files(tmp_path / "vocab.json", tmp_path / "merges.txt")
    source = tmp_path / "plain.txt"
    source.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="uint16"):
        encode_file(tokenizer, source, tmp_path / "ids.bin")


def test_compression_ratio_is_bytes_per_token():
    tokenizer = gpt2_tokenizer()
    documents = ["Once upon a time, there was a café.", "The end."]
    total_bytes = sum(len(d.encode("utf-8")) for d in documents)
    total_tokens = sum(len(tokenizer.encode(d)) for d in documents)
    assert compression_ratio(tokenizer, documents) == total_bytes / total_tokens


def test_sample_documents_is_repeatable_and_drops_empty_documents():
    source = FIXTURES_PATH / "tinystories_sample.txt"
    first = sample_documents(source, 3, seed=0)
    assert first == sample_documents(source, 3, seed=0)
    assert len(first) == 3
    assert all(doc and EOT not in doc for doc in first)
