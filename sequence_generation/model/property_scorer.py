"""
PropertyScorer — ensemble property predictor (proposal: slide 14-15).

Trains K independent SeqRegressor models on the same data with different
random seeds / inits. At inference, the PropertyScorer exposes:

    mu_hat(x)     = mean over K members            (predicted activity)
    sigma2_hat(x) = variance over K members        (epistemic uncertainty)

Used by the guided sampler:
    +alpha * grad mu_hat(x)     -> property guidance
    -beta  * grad sigma2_hat(x) -> activity-cliff / uncertainty penalty
"""

import torch
import torch.nn as nn

from sequence_generation.model.property_regressor import SeqRegressor


class PropertyScorer(nn.Module):
    def __init__(self, alphabet_size, num_members: int = 5,
                 hidden_dim: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_members = num_members
        self.members = nn.ModuleList([
            SeqRegressor(alphabet_size, hidden_dim=hidden_dim,
                         depth=depth, dropout=dropout)
            for _ in range(num_members)
        ])

    def forward(self, batch, mask=None):
        """Returns stacked per-member predictions, shape (K, B, 1)."""
        return torch.stack([m(batch, mask=mask) for m in self.members], dim=0)

    def mu_sigma2(self, batch, mask=None):
        """Returns (mu_hat, sigma2_hat), each shape (B, 1)."""
        preds = self.forward(batch, mask=mask)         # (K, B, 1)
        mu = preds.mean(dim=0)                         # (B, 1)
        var = preds.var(dim=0, unbiased=False)         # (B, 1)
        return mu, var
