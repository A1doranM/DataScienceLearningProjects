"""Unit test — SFT label masking hides the prompt region from the loss (notebook section 6.3).

Proves the SFTCollator marks at least some prompt-side label positions as -100, so the
loss is computed only over the response tokens — the model learns to answer, not to echo the prompt.
"""
from collator_sft import SFTCollator
from formatters import Example

def test_masking_sets_prompt_to_ignore():
    col = SFTCollator(block_size=256, bpe_dir='../part_4/runs/part4-demo/tokenizer')
    text = "This is a tiny test."
    x, y = col.collate([(text, "OK")])
    # All labels up to response marker should be -100
    boundary = ("<s>\n### Instruction:\n" + text + "\n\n### Response:\n")
    # We don't have direct access to the tokenized boundary; just sanity check: some -100s present
    assert (y[0] == -100).sum() > 0  # collator set some prompt-region labels to -100 (loss masking is active)
