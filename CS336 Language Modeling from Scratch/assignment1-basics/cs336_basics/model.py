"""The Transformer language model (handout section 3), built one module at a time.

Every module keeps its weights in nn.Parameter, so model.parameters() finds them and the optimizer (section 4)
can train them.
"""

import math

import torch
from torch import nn


class Linear(nn.Module):
    """y = x Wᵀ: a learned matrix multiply without a bias (handout section 3.3.1, problem `linear`).

    W is stored as (out_features, in_features), the way PyTorch's own nn.Linear stores it.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()  # nn.Module's own setup; without it, assigning an nn.Parameter below fails
        std = math.sqrt(2 / (in_features + out_features))  # the handout's σ: σ² = 2 / (d_in + d_out)
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)  # one row per output
        weight = nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)  # fills it in place
        self.weight = nn.Parameter(weight)  # registered: parameters() and state_dict() now include "weight"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x of shape (..., in_features) -> shape (..., out_features)."""
        return x @ self.weight.T  # each output = one row of W times x
