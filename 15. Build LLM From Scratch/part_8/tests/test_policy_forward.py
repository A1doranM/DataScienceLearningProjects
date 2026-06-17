"""Unit test — PolicyWithValue forward output shapes (notebook section 8.1).

Proves the actor-critic head returns per-token logits (B,T,V) and a scalar value
per token (B,T), so RL training can read both action distribution and state value
from a single forward pass.
"""
import torch
from policy import PolicyWithValue

def test_policy_shapes():
    B,T,V = 2, 16, 256
    pol = PolicyWithValue(vocab_size=V, block_size=T, n_layer=2, n_head=2, n_embd=64)
    x = torch.randint(0, V, (B,T))
    logits, values, loss = pol(x, None)
    assert logits.shape == (B,T,V)  # logits (B,T,V) for actions, values (B,T) for critic
    assert values.shape == (B,T)
