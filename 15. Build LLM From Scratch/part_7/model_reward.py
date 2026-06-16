"""7.2 RewardModel — encoder -> masked-mean-pool -> one scalar reward per sequence.

What this file does
-------------------
Defines the reward model used in RLHF-style preference training. You feed it a
tokenized (prompt + response) sequence and it returns ONE number: how "good"
that response is. Concretely it embeds the tokens, runs them through a small
*bidirectional* Transformer encoder, averages the token vectors into a single
sentence vector (ignoring padding), and projects that down to a scalar:

    tokens (B, T)
        -> tok_emb + pos_emb        ->  h (B, T, C)
        -> TransformerEncoder        ->  h (B, T, C)   (bidirectional, not causal)
        -> LayerNorm                 ->  h (B, T, C)
        -> masked mean over T        ->  pooled (B, C)
        -> Linear(C -> 1)            ->  r (B,)

Two design choices worth flagging:
  * The encoder is BIDIRECTIONAL (plain nn.TransformerEncoder, no causal mask).
    That is fine here because we only *score* a finished sequence, we never
    generate from it, so a token is allowed to "see" tokens to its right.
  * This is a FRESH model with randomly-initialized weights. In real RLHF the
    reward model is usually warm-started from the SFT checkpoint; here we keep
    it simple and train it from scratch.

Where this fits in the Part 7 reward-modeling pipeline
------------------------------------------------------
    preference pair (prompt, chosen, rejected)        (data_prefs.py)
              |
    [ plate both with SFT template + tokenize         (collator_rm.py)  ]
    [   -> (pos_ids, neg_ids)                                           ]
              |
    [ reward model: encoder -> mean-pool -> scalar    (model_reward.py) ]   <-- THIS FILE
    [   r_pos = score(chosen),  r_neg = score(rejected)                 ]
              |
    [ Bradley-Terry loss on the gap                   (loss_reward.py)  ]
    [   softplus(-(r_pos - r_neg))                                      ]
              |
    [ train -> reward checkpoint                      (train_rm.py)     ]
              |
    [ eval: pairwise accuracy r_pos > r_neg           (eval_rm.py)      ]

Math
----
Pooling ignores PAD tokens (PAD id == 2). For one sequence with encoder
outputs h_t (one vector per token t) and per-token keep flag m_t:

    pad_mask_t = 1 if token_t == 2 (PAD) else 0
    m_t        = 1 - pad_mask_t                 (1 for real tokens, 0 for PAD)
    pooled     = sum_t (h_t * m_t) / max(sum_t m_t, 1)
    r          = head(pooled)                   (a single scalar, shape (B,))

  h_t      : encoder output vector for token t, shape (C,)
  m_t      : 1.0 keep / 0.0 drop mask for token t
  sum_t    : sum over the T tokens in the sequence
  max(.,1) : clamp the denominator so an all-PAD row never divides by zero
  head     : Linear(C -> 1) that turns the pooled vector into the reward

Visualization
-------------
See notebook section 7.2 — how a sequence flows from token ids through the
bidirectional encoder and masked-mean pool to a single scalar reward.

Shapes (defaults: n_embd C=256, n_layer=4, n_head=4)
----------------------------------------------------
  input   x        : (B, T)        token ids
  pos             : (1, T)        position indices 0..T-1
  h = emb         : (B, T, C)     token emb + position emb
  h = encoder(h)  : (B, T, C)     bidirectional self-attention
  h = ln(h)       : (B, T, C)     final LayerNorm
  mask            : (B, T, 1)     1.0 for real tokens, 0.0 for PAD (id 2)
  pooled          : (B, C)        masked mean over T
  r = head(pooled): (B,)          one scalar reward per sequence

Parameter count (defaults C=256, V=vocab_size, block=block_size, L=4)
---------------------------------------------------------------------
  tok_emb         : V * C
  pos_emb         : block * C
  encoder layers  : L * (4*C^2 attention + 8*C^2 FFN) ~= L * 12*C^2  (+ small LN/bias terms)
  ln              : 2 * C
  head            : C + 1
"""
from __future__ import annotations
import torch, torch.nn as nn

class RewardModel(nn.Module):
    """Transformer encoder → pooled representation → scalar reward.
    Bidirectional encoder is fine for reward modeling (not used for generation).
    """
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        enc_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,
                                               dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor):
        B, T = x.shape                                          # x: (B, T) token ids
        pos = torch.arange(T, device=x.device).unsqueeze(0)     # (1, T) positions 0..T-1
        h = self.tok_emb(x) + self.pos_emb(pos)                 # (B, T, C) embeddings
        pad_mask = (x == 2)                                     # (B, T) True where token is PAD (id 2)
        h = self.encoder(h, src_key_padding_mask=pad_mask)      # (B, T, C) bidirectional encoder, PADs masked out
        h = self.ln(h)                                          # (B, T, C) final LayerNorm
        # masked mean pool over tokens (ignoring pads)
        mask = (~pad_mask).float().unsqueeze(-1)                # (B, T, 1) 1.0 for real tokens, 0.0 for PAD
        h_sum = (h * mask).sum(dim=1)                           # (B, C) sum of real-token vectors
        len_ = mask.sum(dim=1).clamp_min(1.0)                   # (B, 1) count of real tokens (>=1, never divide by 0)
        pooled = h_sum / len_                                   # (B, C) masked mean = sentence vector
        r = self.head(pooled).squeeze(-1)  # (B,)               # (B,) one scalar reward per sequence
        return r