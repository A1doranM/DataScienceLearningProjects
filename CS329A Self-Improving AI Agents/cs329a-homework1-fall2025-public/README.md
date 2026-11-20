# CS329a Self-Improving AI Agents - Homework 1 (Fall 2025)

Enhanced homework assignment covering test-time compute scaling techniques for self-improving AI agents.

## Overview

This homework explores various techniques to improve model performance during inference, including:

1. Zero-shot Evaluation (10 points)
2. Majority Voting (30 points)
3. Best-of-N with a Generative Reward Model (30 points)
4. Self-Improvement with Feedback (45 points)
5. Analysis (15 points)

## Setup

> **Self-study version.** This copy runs natively on Windows with free NVIDIA build models. The course used conda,
> Python 3.10 and Together AI; its model, Qwen3-Next-80B-A3B-Instruct, has been retired.

1. Create the environment with [uv](https://docs.astral.sh/uv/), from this folder. Use Python 3.10–3.12: on 3.13+,
   `latex2sympy2`'s parser fails to import.

   ```powershell
   uv venv .venv --python 3.12
   uv pip install --python .venv\Scripts\python.exe -e . --config-settings editable_mode=compat
   .venv\Scripts\python.exe -m ipykernel install --user --name cs329a-hw1 --display-name "cs329a-hw1 (Windows)"
   ```

   `editable_mode=compat` matters: `cs329_hw1/` has no `__init__.py`, so a default editable install maps no package.

2. Get a free API key at https://build.nvidia.com and create `.env` in this folder (see `.env.example`) with one of:
   - `NVDA_KEY_FILE=C:/path/to/file.env`: a file containing a line `NVDA_API_KEY=nvapi-...`. Only that line is read.
   - `NVIDIA_NIM_API_KEY=nvapi-...`: the key itself.

   A Hugging Face token (`HUGGINGFACE_HUB_TOKEN`) is optional, because the AIME25 dataset is public.

3. Open `student_homework1.ipynb`, select the kernel **cs329a-hw1 (Windows)**, and run the first cell. It checks the
   kernel and the verifier, sets `MODEL`, and loads the key.

### Troubleshooting

- **`Wrong kernel ...` from the first cell**: select **cs329a-hw1 (Windows)** (Select Kernel → Jupyter Kernel...).
- **`No module named 'typing.io'`**: the kernel runs Python 3.13+. Use the Python 3.12 `.venv`.
- **Accuracy is 0 or suspiciously low**: count responses that start with `Error:`. After 3 failed API calls the
  wrapper returns that text instead of raising, and it scores as a wrong answer.

## Usage

```python
from cs329_hw1.tasks import AIME25
from cs329_hw1.methods import get_sampler, get_verifier

aime25 = AIME25()
problems = aime25.get_problems()
system_prompt = aime25.get_system_prompt()
verifier = get_verifier("aime25")

# Example: sample from the free NVIDIA build model the notebook uses (MODEL)
method = get_sampler("sample_multiple", "nvidia_nim/nvidia/nemotron-3.5-super-vl-120b-a12b", temperature=0.7, n_samples=16, system_prompt=system_prompt)
```

## Features

- **AIME25 Dataset**: Challenging mathematical problems from the American Invitational Mathematics Examination
- **Multiple Sampling Methods**: Greedy, multiple sampling, majority voting, LLM voting
- **Self-Improvement System**: RLEF (Reinforcement Learning from Execution Feedback) with critique and regeneration
- **Comprehensive Analysis**: Detailed comparison of all methods with cost-effectiveness analysis
- **Concurrent Processing**: ThreadPoolExecutor for efficient API calls

## Requirements

- Python 3.10–3.12 (3.12 recommended)
- NVIDIA build API key (free): https://build.nvidia.com
- Hugging Face token (optional)
- Required packages listed in `requirements.txt`

