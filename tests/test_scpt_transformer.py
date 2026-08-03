"""Tests for SCPT transformer policy."""
from __future__ import annotations

import pytest
import torch

from scpt.model.scpt_transformer import SCPTPolicy


def test_forward_shape():
    pol = SCPTPolicy(d=64, pair_dim=14, n_heads=4, n_layers=2)
    L = 16  # 4x4 grid
    P = 3
    z_star = torch.randn(64)
    Z_placed = torch.randn(P, 64)
    F_pair = torch.randn(P, 14)
    grid_xy = torch.randn(L, 2)
    mask = torch.ones(L)
    logits = pol(z_star, Z_placed, F_pair, grid_xy, mask)
    assert logits.shape == (L,)


def test_forward_masks_illegal_cells_to_neg_inf():
    pol = SCPTPolicy(d=64, pair_dim=14, n_heads=4, n_layers=2)
    L = 16
    P = 3
    z_star = torch.randn(64)
    Z_placed = torch.randn(P, 64)
    F_pair = torch.randn(P, 14)
    grid_xy = torch.randn(L, 2)
    mask = torch.ones(L)
    mask[0] = 0.0  # cell 0 illegal
    mask[5] = 0.0  # cell 5 illegal
    logits = pol(z_star, Z_placed, F_pair, grid_xy, mask)
    assert torch.isinf(logits[0]) and logits[0] < 0
    assert torch.isinf(logits[5]) and logits[5] < 0
    # Legal cells should have finite logits
    assert torch.isfinite(logits[1:5]).all()
    assert torch.isfinite(logits[6:]).all()


def test_forward_when_no_placed_components():
    """P=0 → empty context path. Should still produce valid logits."""
    pol = SCPTPolicy(d=64, pair_dim=14, n_heads=4, n_layers=2)
    L = 16
    z_star = torch.randn(64)
    grid_xy = torch.randn(L, 2)
    mask = torch.ones(L)
    Z_placed = torch.zeros(0, 64)
    F_pair = torch.zeros(0, 14)
    logits = pol(z_star, Z_placed, F_pair, grid_xy, mask)
    assert logits.shape == (L,)
    assert torch.isfinite(logits).all()


def test_forward_gradient_flow():
    """Check gradients flow back through the policy (sanity)."""
    pol = SCPTPolicy(d=32, pair_dim=14, n_heads=4, n_layers=1)
    L = 9
    P = 2
    z_star = torch.randn(32, requires_grad=True)
    Z_placed = torch.randn(P, 32)
    F_pair = torch.randn(P, 14)
    grid_xy = torch.randn(L, 2)
    mask = torch.ones(L)
    logits = pol(z_star, Z_placed, F_pair, grid_xy, mask)
    # Pick a legal cell and backprop its logit.
    loss = logits[0]
    loss.backward()
    # z_star should have gradients.
    assert z_star.grad is not None
    assert torch.isfinite(z_star.grad).all()


def test_forward_output_does_not_depend_on_masked_cell_inputs():
    """Changing the grid_xy of an illegal cell should NOT change any logits
    (since the cell's logit gets -inf anyway, and other cells' logits don't
    attend to it — cross-attention is over placed components, not grid cells).
    """
    pol = SCPTPolicy(d=32, pair_dim=14, n_heads=2, n_layers=1)
    torch.manual_seed(0)
    L = 4
    P = 1
    z_star = torch.randn(32)
    Z_placed = torch.randn(P, 32)
    F_pair = torch.randn(P, 14)
    grid_xy_a = torch.randn(L, 2)
    grid_xy_b = grid_xy_a.clone()
    grid_xy_b[0] = torch.tensor([999.0, 999.0])  # huge change to illegal cell
    mask = torch.ones(L)
    mask[0] = 0.0
    logits_a = pol(z_star, Z_placed, F_pair, grid_xy_a, mask)
    logits_b = pol(z_star, Z_placed, F_pair, grid_xy_b, mask)
    # Illegal cell [0] is -inf in both cases. Legal cells should be identical.
    assert torch.allclose(logits_a[1:], logits_b[1:], atol=1e-6)


def test_forward_all_cells_illegal_returns_all_neg_inf():
    pol = SCPTPolicy(d=32, pair_dim=14, n_heads=2, n_layers=1)
    L = 4
    z_star = torch.randn(32)
    Z_placed = torch.randn(1, 32)
    F_pair = torch.randn(1, 14)
    grid_xy = torch.randn(L, 2)
    mask = torch.zeros(L)  # all illegal
    logits = pol(z_star, Z_placed, F_pair, grid_xy, mask)
    assert (logits == float("-inf")).all()
