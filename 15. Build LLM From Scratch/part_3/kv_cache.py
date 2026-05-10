"""3.4 KV cache + Rolling buffer with attention sink.

What this file does
-------------------
Provides two helpers used by `attn_modern.py` during generation:

  KVCache    — a tiny dataclass holding the K and V tensors accumulated so
               far for one attention layer.

  RollingKV  — an alternative cache that *bounds* memory by keeping only
               [first `sink` tokens]  ++  [last `window` tokens].
               This is what enables streaming generation of millions of
               tokens with O(window) memory per layer.

Where this fits in the modern Transformer block
-----------------------------------------------
The KV cache lives *inside* each attention layer — one cache per layer.
During generation the model returns updated caches that we feed back next
step:

    [ ... block 1 ... ] -> KVCache_1
    [ ... block 2 ... ] -> KVCache_2
    [ ... block N ... ] -> KVCache_N

Why a KV cache at all?
----------------------
Self-attention at position t needs Q_t, plus K and V for *all* positions
0..t. Without a cache, generating token t requires re-running the whole
sequence through every layer — O(t) work per token, O(T^2) total to make
T tokens. With a cache, K and V for past tokens are computed once and
re-used: O(1) per new token (per layer), O(T) total. Memory grows with T,
which RollingKV addresses next.

Why a *rolling* cache with attention sink?
------------------------------------------
For very long generations, even the cache becomes huge. The naive fix
"just keep the last `window` tokens" *destroys* quality after the first
crop — empirically the model's attention weights collapse.

The "attention sink" trick (Xiao et al. 2023) noticed that the *first
few* tokens of the sequence absorb a disproportionate share of attention
mass — they act as an anchor. So you keep:

    [ token 0, 1, ... sink-1 ]   ++   [ last `window` tokens ]
       (always present, never crop)        (sliding tail)

Together with the sliding window inside attention itself, this lets the
model run for arbitrarily long contexts with bounded memory.

Math (RollingKV.step)
---------------------
After append:
    K = concat([K_old,  k_new], dim=T)
    V = concat([V_old,  v_new], dim=T)

If len(K) > window + sink, crop:
    K = concat([K[:, :, :sink], K[:, :, -window:]], dim=T)
    V = concat([V[:, :, :sink], V[:, :, -window:]], dim=T)

Visualization
-------------
See notebook section 3.4:
  * timeline showing cache length over generation steps with and without
    RollingKV(window=8, sink=2) — the rolling line plateaus at 10
  * a heatmap of "which positions remain in cache after step t"

Shapes
------
  K, V              : (B, n_kv_head, T_so_far, d_head)
  RollingKV cap     : T_so_far <= window + sink
"""
from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class KVCache:
    k: torch.Tensor  # (B,H,T,D)
    v: torch.Tensor  # (B,H,T,D)

    @property
    def T(self):
        return self.k.size(2)

class RollingKV:
    """Rolling buffer with optional attention sink.
    Keeps first `sink` tokens + last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        # crop
        if self.k.size(2) > self.window + self.sink:
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val, tail_v], dim=2)
        return self.k, self.v