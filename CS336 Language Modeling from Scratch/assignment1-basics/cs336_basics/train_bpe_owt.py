"""Problem train_bpe_expts_owt: train a 32,000-token byte-level BPE on OpenWebText, save it, and report on it.

Run from the assignment folder:  uv run python -m cs336_basics.train_bpe_owt
"""

from pathlib import Path

from cs336_basics.train_bpe_tinystories import train_and_save

DATA = Path.home() / "cs336" / "data" / "owt_train.txt"


def main() -> None:
    train_and_save(DATA, 32_000, "owt")


if __name__ == "__main__":
    main()
