"""6.1 Chat template — wrap an (instruction, response) pair in a fixed text format.

What this file does
-------------------
Supervised fine-tuning (SFT) teaches a base model to answer questions. To do
that, every training example must be turned into ONE flat string that always
looks the same, so the model can learn the pattern "after '### Response:' comes
the answer". This file is that string-formatting step — pure text, no tensors,
no learnable weights.

It defines:
  * ``template``        — the literal format string with fixed marker text.
  * ``Example``         — a tiny dataclass holding one (instruction, response) pair.
  * ``format_example``  — fills BOTH slots; used to build training targets.
  * ``format_prompt_only`` — fills ONLY the instruction and leaves the response
                          slot empty; used at inference, so the model continues
                          the text and writes the answer itself.

The template, in ASCII (the literal markers are shown in quotes):
    "<s>"                       <- start-of-sequence marker (literal text)
    "### Instruction:"          <- literal header
    {instruction}               <- filled in by .format()
    (blank line)
    "### Response:"             <- literal header
    {response} "</s>"           <- filled in (empty at inference) + end marker

Note: "<s>", "</s>", "### Instruction:" and "### Response:" are just plain
characters in the string — they are NOT special tokens here. The tokenizer in
Part 6.3 turns them into ordinary token ids like any other text.

Where this fits in the Part 6 SFT pipeline
------------------------------------------
    raw (instruction, response) pair
              |
    [ 6.1 chat template            (formatters.py)   ]   <-- THIS FILE
              |
    [ 6.2 dataset of pairs         (dataset_sft.py)  ]
              |
    [ 6.3 tokenize + MASK prompt   (collator_sft.py) ]   labels = -100 on the question
              |
    [ load Part 4 base checkpoint                    ]
    [ 6.4 curriculum: easy -> hard (curriculum.py)   ]
              |
    [ 6.5 masked-CE training loop  (train_sft.py)    ]
              |
    [ 6.6 sample with the template (sample_sft.py)   ]
              |
    [ 6.7 evaluate EM / token-F1   (evaluate.py)     ]

Visualization
-------------
See notebook section 6.1 — shows a raw (instruction, response) pair before and
after templating, highlighting the literal markers and the two filled slots.

Shapes
------
  Example.instruction      : str (one question)
  Example.response         : str (one answer)
  format_example(ex)       : str  -> full prompt + answer + "</s>" (training)
  format_prompt_only(instr): str  -> prompt with an EMPTY response slot (inference)
"""
from dataclasses import dataclass

# The fixed chat template. {instruction} and {response} are the only two
# .format() slots; everything else ("<s>", headers, "</s>") is literal text.
template = (
    "<s>\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}</s>"
)

@dataclass
class Example:
    instruction: str  # the user's question / task (plain text)
    response: str     # the desired answer (plain text)


def format_example(ex: Example) -> str:
    # Training form: both slots filled, ends with "</s>". .strip() trims stray
    # whitespace so the markers line up consistently across examples.
    return template.format(instruction=ex.instruction.strip(), response=ex.response.strip())


def format_prompt_only(instruction: str) -> str:
    # Inference form: response slot left empty so the model continues the text
    # after "### Response:" and generates the answer itself.
    return template.format(instruction=instruction.strip(), response="")