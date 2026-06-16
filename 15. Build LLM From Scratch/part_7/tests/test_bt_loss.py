"""Unit test — Bradley-Terry reward loss responds correctly to preference margin (notebook section 7.3).

Proves the loss is monotonically decreasing in the chosen-vs-rejected reward gap: a wider margin yields a lower loss. This is the core training signal that pushes the reward model to score preferred responses higher than rejected ones.
"""
import torch
from loss_reward import bradley_terry_loss

def test_bradley_terry_monotonic():
    pos = torch.tensor([2.0, 3.0])
    neg = torch.tensor([1.0, 1.5])
    l1 = bradley_terry_loss(pos, neg)
    l2 = bradley_terry_loss(pos+1.0, neg)  # increase margin
    assert l2 < l1  # widening the gap (pos+1) lowers the loss -> loss is monotonic in the margin