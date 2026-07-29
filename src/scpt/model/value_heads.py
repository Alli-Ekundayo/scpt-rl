"""Value heads for PPO-EAL: 1 reward critic + N constraint critics.

All critics share the same input embedding (`z_comp`, typically the mean-
pooled GNN output over the design graph) but use SEPARATE linear heads.

Why not a shared MLP? Constraint critics shouldn't fight the reward
critic's representation — separate heads let each critic learn its own
projection of z without gradient interference. Empirically (per the spec's
diagnostic §3.5), if early-quartile critic loss is materially worse than
late-quartile, that's the signal to promote the deferred `[GRAPH]` virtual
node rather than fiddling with head sharing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueHeads(nn.Module):
    """1 reward critic + N constraint critics over a shared z_comp.

    Args:
        d: dimension of the input embedding z_comp (last dim).
        constraint_names: ordered list of constraint names matching the keys
            in the `phi_c_by_name` dict passed to `MomentumDualUpdater`.
    """
    def __init__(self, d: int, constraint_names: list[str]):
        super().__init__()
        self.constraint_names = list(constraint_names)
        self.reward_critic = nn.Linear(d, 1)
        self.constraint_critics = nn.ModuleDict({
            name: nn.Linear(d, 1) for name in constraint_names
        })

    def forward(self, z_comp: torch.Tensor) -> dict[str, torch.Tensor]:
        """Graph-level readout (mean-pool v1) → separate critic values.

        Args:
            z_comp: (N, d) per-node embeddings from the GNN encoder.
                    v1 uses mean-pooling; a future `[GRAPH]` virtual node
                    would bypass pooling.

        Returns:
            dict mapping "reward" + each constraint name → scalar tensor.
        """
        g = z_comp.mean(dim=0)  # (d,) graph-level readout
        out = {"reward": self.reward_critic(g).squeeze(-1)}
        for name, head in self.constraint_critics.items():
            out[name] = head(g).squeeze(-1)
        return out
