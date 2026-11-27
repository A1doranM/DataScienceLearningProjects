"""Problem train_bpe_tinystories: train a 10,000-token byte-level BPE on TinyStories, save it, and report on it.

Run from the assignment folder:  uv run python -m cs336_basics.train_bpe_tinystories
"""

import threading
import time
from pathlib import Path

import psutil

from cs336_basics.bpe import save_vocab_and_merges, train_bpe

DATA = Path.home() / "cs336" / "data" / "TinyStoriesV2-GPT4-train.txt"
OUT = Path.home() / "cs336" / "out"


class PeakMemory:
    """Given: samples the memory of this process and its workers every 0.5 s, and keeps the largest total.

    Memory is counted as PSS (proportional set size): a page that several processes share is split between them,
    so the total counts each page once.
    """

    def __init__(self) -> None:
        self.peak = 0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        me = psutil.Process()
        while not self._done.wait(0.5):
            total = 0
            for proc in [me, *me.children(recursive=True)]:
                try:
                    total += proc.memory_full_info().pss
                except psutil.NoSuchProcess:
                    pass  # a worker that ended between the listing and the reading
            self.peak = max(self.peak, total)

    def stop(self) -> int:
        """Stop sampling and return the peak, in bytes."""
        self._done.set()
        self._thread.join()
        return self.peak


def train_and_save(data: Path, vocab_size: int, name: str) -> None:
    """Train on `data`, save `<name>_vocab.json` and `<name>_merges.txt` in OUT, and print time, memory and the
    longest tokens. Also used by `train_bpe_owt` (problem train_bpe_expts_owt)."""
    memory = PeakMemory()
    start = time.perf_counter()
    vocab, merges = train_bpe(data, vocab_size, ["<|endoftext|>"])
    seconds = time.perf_counter() - start
    peak = memory.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    save_vocab_and_merges(vocab, merges, OUT / f"{name}_vocab.json", OUT / f"{name}_merges.txt")
    learned = [first + second for first, second in merges]
    longest_length = max(len(token) for token in learned)
    longest = [token for token in learned if len(token) == longest_length]  # several can tie
    print(f"trained in {seconds:.1f} s, peak memory {peak / 1e9:.2f} GB")
    print(f"{len(vocab):,} tokens, {len(merges):,} merges, saved to {OUT}")
    print(f"longest tokens ({longest_length} bytes): {longest}")


def main() -> None:
    train_and_save(DATA, 10_000, "tinystories")


if __name__ == "__main__":
    main()
