"""SCPT transformer: sparse coupled placement policy.

Cross-attention over grid cells (queries) and placed components (keys/values).

Why cross-attention instead of self-attention over a merged sequence?
- Grid cells (L) are roughly fixed per step; placed components (P) grow
  monotonically across the episode. Self-attention over the merged
  sequence would need padding + type embeddings and compute attention
  within the grid-cell block that carries no signal.
- Cross-attention matches the problem asymmetry: "where should c* go" is
  a query-to-context lookup.

"Sparse" refers to K/V being only over *placed* components, not all N.
"Coupled" refers to the live-pair features encoding net connectivity
between c* and each placed component.

Output: logits over all H*W grid cells. Illegal cells get `-inf` BEFORE
the Categorical is constructed upstream, matching the PolicyNetwork
convention in the spec.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCPTPolicy(nn.Module):
    """Sparse Coupled Placement Transformer.

    Args:
        d: hidden dimension for queries, keys, values, and output.
        pair_dim: dimension of the live-pair feature vector per placed
            component (14 in the spec).
        n_heads: number of cross-attention heads.
        n_layers: number of cross-attention layers.
        grid_spatial_dim: dimension of the spatial embedding for grid cells
            (the grid_xy tensor is 2D; this projects it to d).
    """
    def __init__(
        self,
        d: int = 256,
        pair_dim: int = 14,
        n_heads: int = 8,
        n_layers: int = 4,
        grid_spatial_dim: int = 2,
    ):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Project query-side inputs to d.
        # z_star: (d,) per-component embedding of the active component.
        # grid_xy: (grid_spatial_dim,) per-cell spatial coordinate.
        # Combined query input: concat(z_star, grid_xy) → (d + grid_spatial_dim,)
        self.query_proj = nn.Linear(d + grid_spatial_dim, d)

        # Project key/value-side inputs to d.
        # For each placed component: concat(z_placed, f_pair) → (d + pair_dim,)
        self.kv_proj = nn.Linear(d + pair_dim, d * 2)  # outputs K and V concatenated

        # Cross-attention layers.
        self.attn_layers = nn.ModuleList([
            _CrossAttentionLayer(d=d, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        # Final projection to logit over grid cells.
        self.logit_head = nn.Linear(d, 1)

        # Learnable empty-context embedding used when P=0 (no placed components
        # yet). Without this, the cross-attention has no K/V to attend to and
        # the policy would just output a constant. The empty-context embedding
        # gives the network something to learn from on the first step.
        self.empty_context = nn.Parameter(torch.randn(d) * 0.02)

    def forward(
        self,
        z_star: torch.Tensor,
        Z_placed: torch.Tensor,
        F_pair: torch.Tensor,
        grid_xy: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits over grid cells.

        Args:
            z_star: (d,) embedding of the active component from GNN encoder.
            Z_placed: (P, d) embeddings of placed components.
            F_pair: (P, pair_dim) live pair features between c* and each placed.
            grid_xy: (L, grid_spatial_dim) spatial coordinates of each grid cell.
            action_mask: (L,) float mask: 1.0 = legal, 0.0 = illegal.

        Returns:
            logits: (L,) — logits over grid cells. Illegal cells get -inf.
        """
        L = grid_xy.shape[0]
        P = Z_placed.shape[0]

        # Broadcast z_star to all L queries: (L, d + grid_spatial_dim)
        z_star_broadcast = z_star.unsqueeze(0).expand(L, -1)
        query_input = torch.cat([z_star_broadcast, grid_xy], dim=-1)
        Q = self.query_proj(query_input)  # (L, d)

        # K, V from placed components.
        if P == 0:
            # Empty context: use learnable embedding as a single key/value.
            # All L queries attend to the same "empty context" embedding.
            empty_ctx = self.empty_context.unsqueeze(0)  # (1, d)
            K = empty_ctx  # (1, d)
            V = empty_ctx  # (1, d)
        else:
            kv_input = torch.cat([Z_placed, F_pair], dim=-1)  # (P, d + pair_dim)
            KV = self.kv_proj(kv_input)  # (P, 2d)
            K, V = KV.split(self.d, dim=-1)  # each (P, d)

        # Apply cross-attention layers.
        H = Q  # (L, d)
        for layer in self.attn_layers:
            H = layer(H, K, V)

        # Final projection: (L, d) → (L, 1) → (L,)
        logits = self.logit_head(H).squeeze(-1)

        # Apply action mask: illegal cells → -inf (so softmax is zero there).
        # Caller constructs the Categorical after this masking.
        illegal = action_mask < 0.5
        logits = logits.masked_fill(illegal, float("-inf"))

        return logits


class _CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with residual connection + LayerNorm."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        # Tiny FFN after attention (standard transformer practice).
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attention: Q queries attend to K/V keys.

        Args:
            Q: (L, d) query embeddings (from grid cells).
            K: (P, d) key embeddings (from placed components).
            V: (P, d) value embeddings (from placed components).

        Returns:
            (L, d) updated query embeddings.
        """
        L, P = Q.shape[0], K.shape[0]
        # Multi-head reshape: Q → (n_heads, L, head_dim); K,V → (n_heads, P, head_dim)
        Q_r = self.q_proj(Q).view(L, self.n_heads, self.head_dim).transpose(0, 1)
        K_r = self.k_proj(K).view(P, self.n_heads, self.head_dim).transpose(0, 1)
        V_r = self.v_proj(V).view(P, self.n_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention: (n_heads, L, P)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q_r, K_r.transpose(-2, -1)) / scale
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, V_r)  # (n_heads, L, head_dim)
        attn_out = attn_out.transpose(0, 1).contiguous().view(L, self.d)
        attn_out = self.out_proj(attn_out)

        # Residual + LayerNorm.
        H = self.norm1(Q + attn_out)
        # FFN residual.
        H = self.norm2(H + self.ffn(H))
        return H
