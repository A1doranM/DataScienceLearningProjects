"""Add rich top-of-file docstrings to each part_1/*.py file."""
import os

PART_1 = r"c:\Home\Projects\DataScienceLearningProjects\15. Build LLM From Scratch\part_1"

# ─────────────────────────────────────────────────────────────────────────────
# The master diagram (as ASCII) used at the top of every file, with a specific
# section highlighted using arrows/brackets.
# ─────────────────────────────────────────────────────────────────────────────

def diagram(highlight: str) -> str:
    """Return the master Transformer-block diagram with a marker next to the
    highlighted node. `highlight` is one of: pe, ln, mha, ffn, residual, mask, all, util.
    """
    boxes = {
        "input":    "  [ Input tokens (B, T, d_model) ]",
        "pe":       "  [ 1.1 Positional Encoding      ]",
        "ln1":      "  [ 1.5 LayerNorm 1              ]",
        "mha":      "  [ 1.3/1.4 Multi-Head Attention ]",
        "add1":     "  [ + residual                   ]",
        "ln2":      "  [ 1.5 LayerNorm 2              ]",
        "ffn":      "  [ 1.5 Feed-Forward             ]",
        "add2":     "  [ + residual                   ]",
        "output":   "  [ Block output (B, T, d_model) ]",
    }
    arrow_down = "                 |"
    arrow_res_r = "                 |---.     (residual skip)"
    arrow_res_l = "                 |<--'"

    # Determine marker per node
    marker_map = {
        "input":  False, "pe": False, "ln1": False, "mha": False,
        "add1":   False, "ln2": False, "ffn": False, "add2": False, "output": False,
    }
    if highlight == "pe":
        marker_map["pe"] = True
    elif highlight == "ln":
        marker_map["ln1"] = True
        marker_map["ln2"] = True
    elif highlight == "mha":
        marker_map["mha"] = True
    elif highlight == "mask":
        marker_map["mha"] = True
    elif highlight == "ffn":
        marker_map["ffn"] = True
    elif highlight == "residual":
        marker_map["add1"] = True
        marker_map["add2"] = True
    elif highlight == "all":
        marker_map = {k: True for k in marker_map}

    def render(key):
        line = boxes[key]
        return ("==>" + line[3:]) if marker_map[key] else line

    lines = [
        render("input"),
        arrow_down,
        render("pe"),
        arrow_down,
        render("ln1"),
        arrow_down,
        render("mha"),
        arrow_down,
        render("add1"),
        arrow_down,
        render("ln2"),
        arrow_down,
        render("ffn"),
        arrow_down,
        render("add2"),
        arrow_down,
        render("output"),
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Docstrings per file
# ─────────────────────────────────────────────────────────────────────────────

DOCSTRINGS = {}

# ────── pos_encoding.py ──────
DOCSTRINGS["pos_encoding.py"] = f'''"""1.1 Positional Encoding — learned and sinusoidal variants.

What this file does
-------------------
Implements two ways to inject position information into token embeddings:
  * LearnedPositionalEncoding    — trainable nn.Embedding(max_len, d_model) table
  * SinusoidalPositionalEncoding — fixed sin/cos waves at geometric frequencies

Self-attention is permutation-invariant (it sees tokens as a *set*), so the
model cannot distinguish "cat sat" from "sat cat" without extra help. The
positional encoding is that extra help: a vector added to each token embedding
that encodes its position index t = 0, 1, 2, ....

Where this fits in the Transformer block
----------------------------------------
{diagram("pe")}

Math (sinusoidal variant)
-------------------------
For position `pos` and dimension index `i`:

    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

  * Even-indexed dims use sine, odd-indexed dims use cosine.
  * The denominator 10000^(2i/d_model) gives each dim its own wavelength:
    low-index dims wiggle fast (sensitive to small position differences),
    high-index dims wiggle slowly (sensitive to long-range differences).
  * Property: dot(PE(pos), PE(pos+k)) depends only on k — the model can
    reason about *relative* offsets.

Visualization
-------------
In the walkthrough notebook (part_1_walkthrough.ipynb, section 1.1):
  * Heatmap of the (max_len, d_model) PE matrix — the classic striped pattern.
  * Line plot of dims 0/4/16/32 — showing frequency decreases with dim index.

Shapes
------
  input  x      : (B, T, d_model)
  PE table      : (max_len, d_model)
  output x + pe : (B, T, d_model)   — PE is added, not concatenated.
"""
'''

# ────── attn_mask.py ──────
DOCSTRINGS["attn_mask.py"] = f'''"""Causal mask helper for autoregressive self-attention.

What this file does
-------------------
Builds the upper-triangular boolean mask that prevents token at position t
from attending to tokens at positions t+1, t+2, ... (the future). This is
the "causal" (a.k.a. decoder-only, autoregressive) pattern used by GPT-style
language models — every token may look at itself and earlier tokens only.

Where this fits in the Transformer block
----------------------------------------
The mask is applied inside the Multi-Head Attention block, after computing
raw scores and before softmax.

{diagram("mask")}

Math
----
Given sequence length T, the mask M is the (T, T) matrix with

    M[i, j] = True   if j > i    (future — blocked)
    M[i, j] = False  if j <= i   (past or self — allowed)

Applied to the attention scores S before softmax:

    S[i, j] = -infinity   where M[i, j] is True
    W = softmax(S)          — the -inf positions become exactly 0

For T=5 the mask looks like (X = blocked, . = allowed):
    .  X  X  X  X
    .  .  X  X  X
    .  .  .  X  X
    .  .  .  .  X
    .  .  .  .  .

Visualization
-------------
See notebook section 1.3 — the mask is shown as a red/white heatmap with
X and . markers on each cell.

Shapes
------
  return   : (1, 1, T, T)   — broadcasts with (B, heads, T, T) attention scores.
"""
'''

# ────── attn_numpy_demo.py ──────
DOCSTRINGS["attn_numpy_demo.py"] = f'''"""1.2 Self-attention from first principles — tiny NumPy example.

What this file does
-------------------
Runs scaled dot-product attention on a hand-picked toy input (T=3, d_model=4,
d_k=2, single head) with fixed W_q, W_k, W_v — no learnable parameters, no
PyTorch. Prints Q, K, V, attention scores, causal-masked weights, and the
final output so you can follow every multiplication on paper.

Where this fits in the Transformer block
----------------------------------------
{diagram("mha")}

Math — scaled dot-product attention
-----------------------------------
    Q = X @ W_q        K = X @ W_k        V = X @ W_v
    S = Q @ K^T / sqrt(d_k)                          # raw similarity scores
    S[i, j] = -infinity for j > i                    # causal mask
    W = softmax(S, dim=-1)                           # row-stochastic weights
    Output = W @ V                                   # weighted values

Intuition — the database analogy
--------------------------------
    Q : "what am I looking for?"    (one query vector per token)
    K : "what do I contain?"        (one key vector per token)
    V : "what will I share?"        (one value vector per token)

For each query row, we compute its dot product with every key row (similarity),
scale down by sqrt(d_k) so variances don't blow up, mask out future positions,
softmax-normalize to get attention weights, then take the corresponding
weighted sum of the value rows.

Visualization
-------------
See notebook section 1.2 — a 4-panel figure showing:
  scores  -->  after causal mask (with -inf above diagonal)
          -->  after softmax (row-stochastic triangular pattern)
          -->  final output = weights @ V
Each cell has the actual number printed on top.

Shapes (this script)
--------------------
  X:           (1, 3, 4)
  W_q, W_k, W_v: each (4, 2)
  Q, K, V:     each (1, 3, 2)
  scores:      (1, 3, 3)   = Q @ K^T / sqrt(d_k)
  weights:     (1, 3, 3)   = softmax(scores with causal mask)
  output:      (1, 3, 2)   = weights @ V
"""
'''

# ────── single_head.py ──────
DOCSTRINGS["single_head.py"] = f'''"""1.3 Single-head self-attention as an nn.Module (PyTorch).

What this file does
-------------------
Same math as attn_numpy_demo.py, but wrapped in an nn.Module so the W_q, W_k,
W_v projection matrices are *learned* via backprop. Returns both the output
and the attention weights (so we can inspect what the head is doing).

Where this fits in the Transformer block
----------------------------------------
This IS the attention mechanism — still a single head; the multi-head version
that a real Transformer uses lives in multi_head.py.

{diagram("mha")}

Math — identical to section 1.2
-------------------------------
    q = x @ W_q       k = x @ W_k       v = x @ W_v       (nn.Linear w/o bias)
    attn = q @ k^T / sqrt(d_k)
    attn = attn.masked_fill(causal_mask, -infinity)       (hide future)
    w    = softmax(attn, dim=-1)
    out  = w @ v                                          (weighted values)

Three PyTorch-specific differences from the NumPy demo
------------------------------------------------------
  1. nn.Linear(d_model, d_k, bias=False) creates the three trainable matrices.
  2. masked_fill applies the causal mask cleanly in one call.
  3. Returns (output, attention_weights) — the weights are useful for viz.

Visualization
-------------
See notebook section 1.3 — the causal mask is shown as a red/white heatmap
with X / . markers for blocked / allowed positions.

Shapes
------
  input  x      : (B, T, d_model)
  q, k, v       : each (B, T, d_k)
  attn scores   : (B, T, T)
  weights       : (B, T, T)   — row-stochastic, lower-triangular
  output        : (B, T, d_k)

Parameter count
---------------
  3 * (d_model * d_k) — three weight matrices (no biases).
"""
'''

# ────── multi_head.py ──────
DOCSTRINGS["multi_head.py"] = f'''"""1.4 Multi-head self-attention with explicit shape tracing.

What this file does
-------------------
Runs n_head independent attention heads in parallel over the same input, each
operating in a d_head = d_model / n_head subspace, then concatenates their
outputs and applies a final learned projection. A single combined W_qkv of
shape (d_model, 3*d_model) replaces three separate matrices — identical math,
faster on GPU.

Where this fits in the Transformer block
----------------------------------------
{diagram("mha")}

Why multiple heads?
-------------------
A single attention head can only learn ONE pattern of "who should attend to
whom". But language has many patterns happening simultaneously:
  * head A might learn "attend to the previous word"   (local context)
  * head B might learn "attend to the subject"          (syntactic)
  * head C might learn "attend to punctuation"          (structural)
Multiple heads let the model learn these in parallel.

Math
----
    qkv = x @ W_qkv                                      # (B, T, 3*d_model)
    split into q, k, v    each reshaped to              # (B, heads, T, d_head)
    attn  = q @ k^T / sqrt(d_head)                      # (B, heads, T, T)
    attn  = attn.masked_fill(causal_mask, -infinity)
    w     = softmax(attn, dim=-1)
    ctx   = w @ v                                       # (B, heads, T, d_head)
    out   = concat heads (transpose+view)  @  W_o        # (B, T, d_model)

Visualization
-------------
See notebook section 1.4:
  * per-head attention heatmaps in a grid (from demo_visualize_multi_head.py)
  * "split heads" visual: one big vector -> rows of colored slices per head

Shapes (d_model=64, n_head=4, d_head=16, T=10)
----------------------------------------------
  input x             : (B, 10, 64)
  after W_qkv         : (B, 10, 192)
  view (B,T,3,H,d_h)  : (B, 10, 3, 4, 16)
  q, k, v (unbind)    : each (B, 10, 4, 16)
  transpose(1,2)      : each (B, 4, 10, 16)
  scores q@k^T/sqrt   : (B, 4, 10, 10)
  weights softmax     : (B, 4, 10, 10)
  ctx = w @ v         : (B, 4, 10, 16)
  merge back          : (B, 10, 64)
  out = merge @ W_o   : (B, 10, 64)

Parameter count
---------------
  W_qkv : d_model * (3 * d_model)   = 3 * d_model^2
  W_o   : d_model * d_model         =     d_model^2
  total : 4 * d_model^2             — the attention parameter budget.
"""
'''

# ────── ffn.py ──────
DOCSTRINGS["ffn.py"] = f'''"""1.5 Feed-Forward Network (FFN / position-wise MLP).

What this file does
-------------------
Applies a small two-layer MLP independently to every token position. The
first Linear expands to a wider hidden dim (mult * d_model, typically 4x),
GELU applies a smooth non-linearity, the second Linear projects back down
to d_model. No information is mixed across positions — that's attention's
job.

Think of it as: attention = communication, FFN = computation.

Where this fits in the Transformer block
----------------------------------------
{diagram("ffn")}

Math
----
    h   = x @ W_1 + b_1            # expand:  d_model    -> 4*d_model
    h   = GELU(h)                  # smooth non-linearity
    out = h @ W_2 + b_2            # project: 4*d_model  -> d_model
    out = Dropout(out)

GELU (Gaussian Error Linear Unit):
    GELU(x) = x * Phi(x)        where Phi is the standard normal CDF
  Behaves like ReLU for x >> 0, ~0 for x << 0, smoothly curved around 0
  (no sharp kink). Its continuous derivative helps gradient flow.

Why expand 4x?
--------------
The expansion gives the network more representational capacity to learn
complex non-linear transformations. The FFN typically holds ~2/3 of all
parameters in a Transformer — it is where most of the "thinking" happens.

Visualization
-------------
See notebook section 1.5a — GELU vs ReLU plotted on the same axes to show
the smooth-vs-kinked difference near zero.

Shapes
------
  input     : (B, T, d_model)
  hidden    : (B, T, mult * d_model)
  output    : (B, T, d_model)     — same shape as input

Parameter count (mult=4)
------------------------
  W_1, b_1 : d_model * 4*d_model + 4*d_model
  W_2, b_2 : 4*d_model * d_model + d_model
  total    : ~8 * d_model^2   — dominant parameter cost of the block.
"""
'''

# ────── block.py ──────
DOCSTRINGS["block.py"] = f'''"""1.6 Transformer block — LN -> MHA -> + -> LN -> FFN -> +.

What this file does
-------------------
Composes LayerNorm, Multi-Head Attention, LayerNorm, and FeedForward with
two residual skip connections into a single repeatable unit. A full GPT
stacks N of these blocks (N = 6 / 12 / 24 / 96 depending on model size).

Where this fits in the Transformer block
----------------------------------------
The whole block — every node is active here:

{diagram("all")}

The forward pass in two lines
-----------------------------
    x = x + self.attn(self.ln1(x))[0]      # communication (tokens exchange info)
    x = x + self.ffn (self.ln2(x))         # computation   (each token processes)

Math — Pre-Norm pattern
-----------------------
    x_1 = x    + MHA(LayerNorm(x))
    x_2 = x_1  + FFN(LayerNorm(x_1))
    output = x_2

Two crucial design choices:

  1. Pre-Norm: LayerNorm is applied BEFORE each sublayer, not after. Makes
     training much more stable than the original Post-Norm paper. The
     residual path stays "clean" — raw x is added without normalization
     in between.

  2. Residual connections: `x + sublayer(...)` provides a gradient highway.
     Even if the sublayer attenuates gradients heavily, the direct skip
     keeps the signal alive — essential for training deep stacks.

LayerNorm
---------
Normalizes each token independently (across its d_model features) to zero
mean and unit variance, then scales and shifts by learnable gamma, beta:

    LN(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
    where mu, sigma^2 are computed across the last (feature) dimension.

Unlike BatchNorm, it doesn't depend on batch statistics — same behavior
at train and inference time.

Visualization
-------------
See notebook section 1.6 — full shape trace line by line through one
forward pass, plus parameter count broken down by submodule.

Shapes
------
  input x  : (B, T, d_model)
  output   : (B, T, d_model)     — block preserves shape (so it can be stacked)
"""
'''

# ────── vis_utils.py ──────
DOCSTRINGS["vis_utils.py"] = f'''"""Plotting helpers for Part 1 visualizations.

What this file does
-------------------
Two small helpers used by the demo scripts:

  * save_matrix_heatmap(mat, title, filename, ...)
      Saves a 2D array as a colored heatmap with a title, axis labels and
      colorbar. Used for PE heatmaps, attention score matrices, etc.

  * save_attention_heads_grid(weights, filename, title_prefix="Head")
      Given a (1, H, T, T) tensor of attention weights, saves all H heads
      in a single grid figure — one subplot per head, labeled "Head 0",
      "Head 1", .... Used by demo_visualize_multi_head.py.

All outputs are written to  part_1/out/  (directory is created if missing).

Where this fits in the Transformer block
----------------------------------------
These are visualization utilities — they don't sit inside the block
itself. They're called by the demo scripts and the walkthrough notebook
to render matrices and attention maps.

Design notes
------------
  * Uses Matplotlib defaults (no custom colormap or style) so outputs are
    readable on any system.
  * Figures are closed after saving to avoid memory buildup in loops.
"""
'''

# ────── demo_mha_shapes.py ──────
DOCSTRINGS["demo_mha_shapes.py"] = f'''"""Walkthrough: prints every MHA intermediate tensor's shape, step by step.

What this file does
-------------------
Runs one forward pass through MultiHeadSelfAttention with B=1, T=5, d_model=12,
n_head=3, and prints the shape of every intermediate tensor — before and after
the reshape / transpose / matmul / softmax / merge steps. Also writes the log
to part_1/out/mha_shapes.txt so you can re-read it later.

Use this as a checklist when reading multi_head.py — trace each print line
back to the source to verify you understand the reshapes.

Where this fits in the Transformer block
----------------------------------------
Exercises the MHA block specifically:

{diagram("mha")}

What you'll see printed
-----------------------
    Input x:           (1, 5, 12)     = (B, T, d_model)
    Linear qkv(x):     (1, 5, 36)     = (B, T, 3*d_model)
    view to 5D:        (1, 5, 3, 3, 4)= (B, T, 3, heads, d_head)
    q,k,v split:       each (1, 5, 3, 4)
    transpose heads:   each (1, 3, 5, 4) = (B, heads, T, d_head)
    scores q@k^T:      (1, 3, 5, 5)   = (B, heads, T, T)
    softmax(weights):  (1, 3, 5, 5)
    context @v:        (1, 3, 5, 4)   = (B, heads, T, d_head)
    merge heads:       (1, 5, 12)     = (B, T, d_model)
    final proj:        (1, 5, 12)     = (B, T, d_model)

How to run
----------
    cd part_1
    python demo_mha_shapes.py
"""
'''

# ────── demo_visualize_multi_head.py ──────
DOCSTRINGS["demo_visualize_multi_head.py"] = f'''"""Visualize multi-head attention weights per head (grid).

What this file does
-------------------
Runs a forward pass through MultiHeadSelfAttention (B=1, T=5, d_model=12,
n_head=3) on a random input, then saves the per-head (T, T) attention
weight matrices as a single grid figure to  part_1/out/multi_head_attn_grid.png.

Each subplot in the grid corresponds to one attention head and shows the
(query_pos, key_pos) attention map — darker = more attention. Even with
random initialization, different heads produce different patterns; after
training on real text, these patterns become meaningful (local, syntactic,
punctuation-attending, etc.).

Where this fits in the Transformer block
----------------------------------------
{diagram("mha")}

What the output image shows
---------------------------
  * rows  = query positions (who's looking)
  * cols  = key positions (what's being looked at)
  * Because the MHA applies a causal mask, the attention weights are
    lower-triangular: upper-right of each heatmap is effectively zero.

How to run
----------
    cd part_1
    python demo_visualize_multi_head.py
"""
'''

# ────── orchestrator.py ──────
DOCSTRINGS["orchestrator.py"] = f'''"""Part 1 orchestrator — runs tests and optional visualizations.

What this file does
-------------------
Driver script for Part 1. Runs, in order:
  1. attn_numpy_demo.py           (sanity-check the attention math on toy numbers)
  2. pytest tests/test_attn_math.py     (NumPy math == PyTorch single-head)
  3. pytest tests/test_causal_mask.py   (mask shape and values)
  4. demo_mha_shapes.py           (print every MHA intermediate shape)
  5. demo_visualize_multi_head.py (only with --visualize — saves PNGs to ./out/)

All sub-scripts are invoked with their working directory set to part_1/ so
their local imports (from multi_head import ..., from attn_mask import ...)
resolve correctly.

Where this fits in the Transformer block
----------------------------------------
Orchestrator doesn't implement block logic itself — it exercises the
pieces that the other files define. Conceptually it drives the whole
block end-to-end:

{diagram("all")}

How to run
----------
    cd part_1
    python orchestrator.py              # tests + shape demo
    python orchestrator.py --visualize  # also save attention heatmap PNGs

Where output lives
------------------
    part_1/out/   — images and logs from the visualization / shape demos.
"""
'''

# ─────────────────────────────────────────────────────────────────────────────
# Apply docstrings to files
# ─────────────────────────────────────────────────────────────────────────────

def strip_existing_docstring(src: str) -> str:
    """Remove a top-of-file module docstring if present."""
    s = src.lstrip()
    # Does the file start with a triple-quoted string?
    for q in ('"""', "'''"):
        if s.startswith(q):
            end = s.find(q, len(q))
            if end == -1:
                return src  # unterminated — leave alone
            # Skip past the closing quotes and any trailing newlines
            after = s[end + len(q):]
            # Preserve leading whitespace that was stripped
            leading = src[: len(src) - len(s)]
            return leading + after.lstrip("\n")
    return src


for fname, docstring in DOCSTRINGS.items():
    path = os.path.join(PART_1, fname)
    if not os.path.exists(path):
        print(f"  SKIP (missing): {fname}")
        continue
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    stripped = strip_existing_docstring(original)
    new_content = docstring + "\n" + stripped.lstrip()
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"  Wrote: {fname}  ({len(docstring.splitlines())} lines of docstring)")

print("\nDone.")
