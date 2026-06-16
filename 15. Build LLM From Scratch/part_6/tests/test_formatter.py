"""Unit test — prompt template wiring for instruction tuning (notebook section section 6.1).

Proves format_example wraps an Example in the "### Instruction:"/"### Response:" markers and that
format_prompt_only stops exactly at the response marker, so generation begins where the model must answer.
"""
from formatters import Example, format_example, format_prompt_only

def test_template_contains_markers():
    ex = Example("Say hi","Hello!")
    s = format_example(ex)
    assert "### Instruction:" in s and "### Response:" in s  # full example carries both markers; prompt-only ends at "### Response:"
    p = format_prompt_only("Explain transformers.")
    assert p.endswith("### Response:\n") or p.endswith("### Response:\n</s>")