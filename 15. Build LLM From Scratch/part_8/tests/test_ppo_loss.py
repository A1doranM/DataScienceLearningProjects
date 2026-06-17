"""Unit test — PPO clipped objective returns a scalar loss (notebook section section 8.5).

Proves ppo_losses collapses a per-token batch (ratio=1.2, adv=1, clip_ratio=0.1) into a
single 0-dim total_loss, the property that lets the optimizer call .backward() on it.
"""
import torch
from ppo_loss import ppo_losses

def test_clipped_objective_behaves():
    N = 32
    old_logp = torch.zeros(N)
    new_logp = torch.log(torch.full((N,), 1.2))  # ratio=1.2
    adv      = torch.ones(N)
    new_v = torch.zeros(N)
    old_v = torch.zeros(N)
    ret  = torch.ones(N)
    out = ppo_losses(new_logp, old_logp, adv, new_v, old_v, ret, clip_ratio=0.1)
    assert out.total_loss.ndim == 0  # scalar (0-dim): one reduced loss for the whole batch