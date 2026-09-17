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

        # Learnable empty-context embeddings used when P=0 (no placed components
        # yet). Without K/V, the cross-attention has nothing to attend to and
        # the policy would output a constant. Two separate parameters (one for K,
        # one for V) let the network learn distinct lookup-key vs. read-value
        # behaviours for the first placement step.
        self.empty_context_k = nn.Parameter(torch.randn(d) * 0.02)
        self.empty_context_v = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, z_star, Z_placed, F_pair, grid_xy, action_mask, kv_mask=None):
        """Compute logits over grid cells.
        
        Supports both single-sample and batched inputs:
        - Single: z_star (d,), Z_placed (P, d), grid_xy (L, 2), action_mask (L,)
        - Batched: z_star (B, d), Z_placed (B, P, d), grid_xy (B, L, 2), action_mask (B, L)
        
        Args:
            z_star: (d,) embedding of the active component from GNN encoder.
            Z_placed: (P, d) embeddings of placed components.
            F_pair: (P, pair_dim) live pair features between c* and each placed.
            grid_xy: (L, grid_spatial_dim) spatial coordinates of each grid cell.
            action_mask: (L,) float mask: 1.0 = legal, 0.0 = illegal.
            kv_mask: optional (B, P) bool; True = padding in Z_placed/F_pair to ignore.

        Returns:
            logits: (L,) — logits over grid cells. Illegal cells get -inf.
        """
        # Auto-batch single-sample inputs
        unbatched = z_star.dim() == 1
        if unbatched:
            result = self.forward(
                z_star.unsqueeze(0), Z_placed.unsqueeze(0), F_pair.unsqueeze(0),
                grid_xy.unsqueeze(0), action_mask.unsqueeze(0), kv_mask=kv_mask,
            )
            return result.squeeze(0)
        
        B = z_star.shape[0]
        L = grid_xy.shape[1]
        P = Z_placed.shape[1]
        
        z_star_broadcast = z_star.unsqueeze(1).expand(B, L, -1)
        query_input = torch.cat([z_star_broadcast, grid_xy], dim=-1)
        Q = self.query_proj(query_input)
        
        if P == 0:
            # Empty context: separate learnable K and V give the network the
            # ability to learn distinct first-step lookup-key vs. read-value
            # behaviours rather than conflating them into a single vector.
            K = self.empty_context_k.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            V = self.empty_context_v.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            kv_mask = None
        else:
            kv_input = torch.cat([Z_placed, F_pair], dim=-1)
            KV = self.kv_proj(kv_input)
            K, V = KV.split(self.d, dim=-1)
        
        H = Q
        for layer in self.attn_layers:
            H = layer(H, K, V, kv_mask=kv_mask)
        
        logits = self.logit_head(H).squeeze(-1)
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

    def forward(self, Q, K, V, kv_mask=None):
        """Cross-attention: Q queries attend to K/V keys.
        
        Supports both unbatched (L, d) and batched (B, L, d) inputs.
        
        Args:
            Q: (B, L, d) query embeddings.
            K: (B, P, d) key embeddings.
            V: (B, P, d) value embeddings.
            kv_mask: optional (B, P) bool tensor; True = padding position to ignore.
            
        Note on LayerNorm placement:
            This layer uses post-LN style (original "Attention Is All You Need"
            convention): LayerNorm is applied *after* the residual addition.
            Post-LN is stable for up to n_layers ≈ 6; if depth is scaled beyond
            that, switch to pre-LN (apply norm to the residual branch input
            before the sub-layer) to improve gradient flow.
        """
        B, L, _ = Q.shape
        P = K.shape[1]
        Q_r = self.q_proj(Q).view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K_r = self.k_proj(K).view(B, P, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V_r = self.v_proj(V).view(B, P, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q_r, K_r.transpose(-2, -1)) / scale
        if kv_mask is not None:
            scores = scores.masked_fill(kv_mask[:, None, None, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        # Handle NaN from all-masked rows (all -inf → softmax NaN)
        weights = weights.nan_to_num(0.0)
        attn_out = torch.matmul(weights, V_r)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, self.d)
        attn_out = self.out_proj(attn_out)
        
        # Post-LN residual: norm applied after the residual addition.
        H = self.norm1(Q + attn_out)
        # FFN residual (post-LN).
        H = self.norm2(H + self.ffn(H))
        return H
