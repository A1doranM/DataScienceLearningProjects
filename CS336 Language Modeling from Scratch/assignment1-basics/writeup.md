# CS336 Assignment 1: write-up

Answers to the handout's written deliverables, in the handout's order.

## Problem (train_bpe_tinystories): BPE Training on TinyStories

Measured on 2026-11-19 in WSL2 (16 logical CPUs, 15 GB RAM) with `uv run python -m cs336_basics.train_bpe_tinystories`.
The vocabulary and merges were saved to `~/cs336/out/tinystories_vocab.json` and `~/cs336/out/tinystories_merges.txt`.

**(a)** Training on the 2.2 GB TinyStories training set with a 10,000-token vocabulary took 84.6 s and at most 2.76 GB
of memory across all processes. The longest token is `' accomplishment'` (15 bytes, with its leading space). It makes
sense: it's a real English word that appears 1,516 times in the data, typical of the lessons many TinyStories end with
("a sense of accomplishment"), and BPE built it from pieces it had already learned.

**(b)** Pre-tokenization takes the most time: 67% (55.5 s of 83 s), even with 16 processes, because the regex has to
run over all 2.2 GB of text. The merge loop takes the other 33%, and about 95% of that goes to `max()`, which scans all
~38,000 pairs every round to find the most frequent one.
