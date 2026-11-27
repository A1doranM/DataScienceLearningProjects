"""Problem tokenizer_experiments (handout section 2.7): compression ratios, throughput, and whole datasets encoded to
uint16 token IDs for training.

Run from the assignment folder:  uv run python -m cs336_basics.tokenizer_experiments
The encoded datasets are written to ~/cs336/out/<name>_<split>.bin: raw uint16 IDs with no header, read back with
np.memmap(path, dtype=np.uint16, mode="r").
"""

import multiprocessing
import os
import random
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from cs336_basics.bpe import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
DATA = Path.home() / "cs336" / "data"
OUT = Path.home() / "cs336" / "out"
# What one worker encodes at a time. A worker needs about 20 times its chunk in memory while encoding (one 16 MiB
# TinyStories chunk peaked at 336 MB: the list of 4.1 M pieces), so 4 MiB keeps 16 workers small; the IDs do not
# depend on it.
CHUNK_BYTES = 4 * 1024 * 1024
PILE_BYTES = 825e9  # the Pile, the handout's example for part (c)

_worker_tokenizer: Tokenizer | None = None  # each worker process's own copy, set once by _init_worker


def _init_worker(tokenizer: Tokenizer) -> None:
    global _worker_tokenizer
    _worker_tokenizer = tokenizer


def _encode_chunk(job: tuple[str | os.PathLike, int, int]) -> np.ndarray:
    """Encode bytes start..end of the file to uint16 IDs. Runs inside one worker process."""
    input_path, start, end = job
    assert _worker_tokenizer is not None, "_init_worker was not called"
    with open(input_path, "rb") as f:  # bytes, because start and end are byte positions
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return np.array(_worker_tokenizer.encode(text), dtype=np.uint16)


def encode_file(
    tokenizer: Tokenizer, input_path: str | os.PathLike, output_path: str | os.PathLike, chunk_bytes: int = CHUNK_BYTES
) -> int:
    """Encode the whole file at `input_path` and write its IDs to `output_path` as raw uint16 values; return their
    count. The IDs are the same as tokenizer.encode(whole text).

    The file is cut into chunks of about `chunk_bytes`, each cut just before the tokenizer's first special token, so
    no piece spans two chunks; worker processes encode the chunks in parallel, and the IDs are written in file
    order as each chunk finishes, so memory stays flat however big the file is. Raises ValueError when the
    vocabulary has an ID that uint16 cannot hold (above 65,535).
    """
    if max(tokenizer.vocab) >= 2**16:
        raise ValueError(f"the largest ID, {max(tokenizer.vocab)}, does not fit in uint16 (0 to 65,535)")
    file_size = os.path.getsize(input_path)
    if tokenizer.special_tokens:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, max(1, file_size // chunk_bytes), tokenizer.special_tokens[0].encode()
            )
    else:
        boundaries = [0, file_size]
    jobs = [(input_path, start, end) for start, end in pairwise(boundaries)]
    count = 0
    with open(output_path, "wb") as out:
        if len(jobs) <= 1:  # one chunk: encode it right here, without starting worker processes
            _init_worker(tokenizer)
            for job in jobs:
                ids = _encode_chunk(job)
                ids.tofile(out)
                count += ids.size
        else:
            workers = min(os.cpu_count() or 1, len(jobs))
            context = multiprocessing.get_context("forkserver")
            with context.Pool(workers, initializer=_init_worker, initargs=(tokenizer,)) as pool:
                for ids in pool.imap(_encode_chunk, jobs):  # imap keeps the chunks in file order
                    ids.tofile(out)
                    count += ids.size
    return count


def sample_documents(path: str | os.PathLike, n: int, seed: int = 0) -> list[str]:
    """Return `n` documents picked at random (the same ones for the same seed) from a file of documents separated
    by <|endoftext|>."""
    text = Path(path).read_text(encoding="utf-8")
    documents = [doc.strip() for doc in text.split(EOT)]
    documents = [doc for doc in documents if doc]
    return random.Random(seed).sample(documents, n)


def compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> float:
    """Bytes per token over all the documents together: how many bytes of UTF-8 text one token stands for."""
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in documents)
    return total_bytes / total_tokens


def throughput(tokenizer: Tokenizer, text: str) -> float:
    """Bytes per second for tokenizer.encode on `text`, in this one process."""
    start = time.perf_counter()
    tokenizer.encode(text)
    return len(text.encode("utf-8")) / (time.perf_counter() - start)


DATASETS = {  # tokenizer name -> the data files it was trained for
    "tinystories": {"valid": "TinyStoriesV2-GPT4-valid.txt", "train": "TinyStoriesV2-GPT4-train.txt"},
    "owt": {"valid": "owt_valid.txt", "train": "owt_train.txt"},
}


def main() -> None:
    tokenizers: dict[str, Tokenizer] = {}
    samples: dict[str, list[str]] = {}
    for name, files in DATASETS.items():
        vocab_path = OUT / f"{name}_vocab.json"
        if not vocab_path.exists():
            print(f"{name}: no trained tokenizer at {vocab_path}, skipped")
            continue
        tokenizers[name] = Tokenizer.from_files(vocab_path, OUT / f"{name}_merges.txt", [EOT])
        samples[name] = sample_documents(DATA / files["valid"], 10)

    print("== (a)/(b) compression ratio in bytes per token, 10 sampled validation documents")
    for docs_name, documents in samples.items():
        for tok_name, tok in tokenizers.items():
            print(f"  {docs_name} documents, {tok_name} tokenizer: {compression_ratio(tok, documents):.2f}")

    print("== (c) throughput of encode in one process, starting with an empty cache")
    for name, tok in tokenizers.items():
        text = (DATA / DATASETS[name]["valid"]).read_text(encoding="utf-8")[:20_000_000]  # 20 M characters
        rate = throughput(Tokenizer(tok.vocab, list(tok.ranks), tok.special_tokens), text)
        print(f"  {name}: {rate / 1e6:.2f} MB/s; the Pile (825 GB) would take {PILE_BYTES / rate / 86400:.1f} days")

    print("== (d) whole datasets to uint16 IDs")
    for name, tok in tokenizers.items():
        for split, file_name in DATASETS[name].items():
            source = DATA / file_name
            start = time.perf_counter()
            count = encode_file(tok, source, OUT / f"{name}_{split}.bin")
            seconds = time.perf_counter() - start
            size = source.stat().st_size
            print(
                f"  {name} {split}: {size:,} bytes -> {count:,} IDs in {seconds:.0f} s "
                f"({size / seconds / 1e6:.1f} MB/s with up to {os.cpu_count()} processes)"
            )


if __name__ == "__main__":
    main()
