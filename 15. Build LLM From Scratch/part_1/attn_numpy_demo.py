"""1.2 Self-attention from first principles — tiny NumPy example.

What this file does
-------------------
Runs scaled dot-product attention on a hand-picked toy input (T=3, d_model=4,
d_k=2, single head) with fixed W_q, W_k, W_v — no learnable parameters, no
PyTorch. Prints Q, K, V, attention scores, causal-masked weights, and the
final output so you can follow every multiplication on paper.

Where this fits in the Transformer block
----------------------------------------
  [ Input tokens (B, T, d_model) ]
                 |
  [ 1.1 Positional Encoding      ]
                 |
  [ 1.5 LayerNorm 1              ]
                 |
==> 1.3/1.4 Multi-Head Attention ]
                 |
  [ + residual                   ]
                 |
  [ 1.5 LayerNorm 2              ]
                 |
  [ 1.5 Feed-Forward             ]
                 |
  [ + residual                   ]
                 |
  [ Block output (B, T, d_model) ]

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

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Toy inputs (batch=1, seq=3, d_model=4)
X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)

# Weight matrices (learned in real models). We fix numbers for determinism.
Wq = np.array([[ 0.2, -0.1],
               [ 0.0,  0.1],
               [ 0.1,  0.2],
               [-0.1,  0.0]], dtype=np.float32)
Wk = np.array([[ 0.1,  0.1],
               [ 0.0, -0.1],
               [ 0.2,  0.0],
               [ 0.0,  0.2]], dtype=np.float32)
Wv = np.array([[ 0.1,  0.0],
               [-0.1,  0.1],
               [ 0.2, -0.1],
               [ 0.0,  0.2]], dtype=np.float32)

# Project to Q, K, V
Q = X @ Wq  # (1,3,2)
K = X @ Wk  # (1,3,2)
V = X @ Wv  # (1,3,2)

print("Q shape:", Q.shape, "\nQ=\n", Q[0])
print("K shape:", K.shape, "\nK=\n", K[0])
print("V shape:", V.shape, "\nV=\n", V[0])

# Scaled dot-products
scale = 1.0 / np.sqrt(Q.shape[-1])
attn_scores = (Q @ K.transpose(0,2,1)) * scale  # (1,3,3)

# Causal mask (upper triangle set to -inf so softmax->0)
mask = np.triu(np.ones((1,3,3), dtype=bool), k=1)
attn_scores = np.where(mask, -1e9, attn_scores)

# Softmax over last dim
weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)
print("Weights shape:", weights.shape, "\nAttention Weights (causal)=\n", weights[0])

# Weighted sum of V
out = weights @ V  # (1,3,2)
print("Output shape:", out.shape, "\nOutput=\n", out[0])