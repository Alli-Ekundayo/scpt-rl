"""Momentum dual updater for PPO-EAL's Lagrangian.

The dual variables (λ) enforce soft constraints via the augmented-Lagrangian
method. After each PPO update we:
1. Compute the constraint violation `phi_c_i` for each constraint (the
   expected advantage under the current policy, shifted by the budget).
2. Smooth `phi_c_i` with an EMA to damp oscillation.
3. Ascend λ in the direction of the smoothed violation, clamped to [0, ∞).

One `MomentumDualUpdater` instance per training run. State persists across
outer updates — the λ trajectory is accumulated over the whole run, not
reset per iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MomentumDualUpdater:
    """Momentum-smoothed dual ascent for Lagrangian multipliers.

    Attributes:
        constraint_names: ordered list of constraint names (must match the
            keys in `phi_c_by_name` passed to `update`).
        alpha: learning rate for the dual ascent step.
        ema_decay: EMA smoothing coefficient for phi_c. 0 = no smoothing
            (raw phi), 0.99 = very heavy smoothing. 0.9 is a reasonable default.
    """
    constraint_names: list[str]
    alpha: float
    ema_decay: float
    _lambdas: dict[str, float] = field(default_factory=dict)
    _ema: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for n in self.constraint_names:
            self._lambdas.setdefault(n, 0.0)
            self._ema.setdefault(n, 0.0)

    def update(self, phi_c_by_name: dict[str, float]) -> dict[str, float]:
        """Apply one dual-ascent step. Returns the updated λ values.

        Each constraint's update:
            ema_i = decay * ema_i + (1 - decay) * phi_c_i
            lambda_i = max(0, lambda_i + alpha * ema_i)

        The clamping to [0, ∞) enforces dual feasibility — λ for satisfied
        constraints (negative phi_c) decays back toward 0 instead of going
        negative.
        """
        out: dict[str, float] = {}
        for n in self.constraint_names:
            phi = phi_c_by_name.get(n, 0.0)
            self._ema[n] = self.ema_decay * self._ema[n] + (1.0 - self.ema_decay) * phi
            self._lambdas[n] = max(0.0, self._lambdas[n] + self.alpha * self._ema[n])
            out[n] = self._lambdas[n]
        return out

    @property
    def lambdas(self) -> dict[str, float]:
        """Read-only snapshot of current λ values."""
        return dict(self._lambdas)
