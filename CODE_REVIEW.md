# SCPT-RL — Code Review

**Date:** 2026-08-19
**Scope:** all Python source under `src/scpt/`, entry-point scripts in `scripts/`, configs in `configs/`, and pytest suite under `tests/`. Rust crates reviewed at the surface (boundary re-exports only).

**Recent context:** the last 5 commits cluster around (a) constraint-critic training wiring (`2e85dca`, `c0b8f5e`, `a1b608f`, `5ec9b10`), (b) cost normalisation fixes (`f1a84ba`, `f877567`, `c83d19c`, `3a5f9b6`), and (c) diagnostics surfacing (`j_c`/`c_adv_mean`). Most correctness risk in the current tree therefore lives in those recently-touched paths, which is where this review focused most scrutiny.

---

## Findings (most-severe first)

### F1 — CRITICAL: Constraint GAE is *never* masked by `legal_flags`
**File:** `src/scpt/agent/ppo_eal.py:529-561` (the `_compute_constraint_gae` method)

The function computes `legal_flags` from each step's stored mask and the action that was actually taken, then asserts `torch.all(legal_flags == 1.0)`. After that, it computes `raw_gae = compute_gae(costs_t, values, dones, next_value, ...)` and **returns it without ever weighting by `legal_flags`**. The docstring says:

> `constraint_advantages` must already be averaged over LEGAL actions only.

…but this code never does that averaging. Every step's advantage is returned equally, regardless of whether the chosen action was legal. Consequence: the constraint surrogate and the policy-gradient penalty term both pull toward an expectation over the *uniform* policy instead of the *actual* policy, which silently biases `phi_c` (and so `λ`) and biases the linear-Lagrangian policy gradient.

The function name, the assertion, and the docstring together suggest this is a regression where the averaging was either deleted or never written. The `assert` is also a *correctness guard* — if it ever fires (e.g. early-episode illegal action), the trainer will hard-crash the run.

**Suggested fix:**
```python
# weight advantage by legal indicator so illegal-step costs don't leak into phi_c
legal_t = legal_flags.float()
weighted = raw_gae * legal_t
return weighted / legal_t.sum().clamp(min=1.0)
```
…or compute the average *only over legal steps* per spec §5.3.

---

### F2 — CRITICAL: Constraint-policy gradient is summed across constraints, not mean
**File:** `src/scpt/agent/ppo_eal.py:722-727`

```python
constraint_penalty_i = torch.zeros((), device=self.device)
for cname in self.cfg.constraint_names:
    c_adv_std = c_adv_raw.std().clamp(min=1e-6)
    c_adv_i = c_adv_raw[i] / c_adv_std
    constraint_penalty_i = constraint_penalty_i + lambdas[cname] * ratio * c_adv_i
```

With three constraints, the per-step gradient from the policy is **3×** as large as a single-constraint configuration, *before* any lambda scaling. The `constraint_names` list is a config knob that an operator can extend (the YAML comment says "extend this list to add new constraint heads"); silently tripling the gradient magnitude when they add a new head is dangerous. The reward surrogate is `mean()`-reduced in `clipped_ppo_surrogate`, so the two terms are also on different scales.

**Fix:** divide by `len(self.cfg.constraint_names)`, or compute `sum(c) / n`, or wrap in a `torch.stack(...).mean()` call before adding to the surrogate.

---

### F3 — HIGH: `c_adv_std` normaliser uses raw advantages, so it's identical for every sample
**File:** `src/scpt/agent/ppo_eal.py:725-726`

```python
c_adv_std = c_adv_raw.std().clamp(min=1e-6)
c_adv_i = c_adv_raw[i] / c_adv_std
```

`c_adv_std` is computed outside the per-sample loop and is the *same scalar* for every `i` in the minibatch. That's effectively the same as globally normalising `c_advs_dict[cname]` once before the epoch loop — fine for variance control, but the comment promises it "decouples the 100x-amplified scale (1/(1-gamma))". The decoupling works because the *normaliser* is the *same* for every sample in the epoch — correct, just misleadingly written. A minor cleanup is to hoist this out of the per-sample loop (and use `normalize_advantages(constraint_advs_dict[cname])` for symmetry with the reward path), but it's not a bug.

---

### F4 — HIGH: `clip_grad_norm_` runs *after* every backward but before `optimizer.step` — `clip_grad_norm_` is fine, but gradient accumulation is fragile
**File:** `src/scpt/agent/ppo_eal.py:657-758`

The training loop does **one-sample-at-a-time backprop inside a minibatch**, dividing each sample's loss by `batch_size`. That's a manual gradient-accumulation pattern. Two issues:

1. **Memory:** you keep every intermediate `loss_i.backward()` graph live until the optimiser step. For minibatch=64 on a real board you hold 64 separate graphs simultaneously.
2. **Correctness:** `clip_grad_norm_` is computed on the *accumulated* gradients, but the docstring `max_norm=0.5` is small enough that even one outlier sample (e.g. a high-HPWL step) can saturate the entire minibatch's clip — and the per-sample gradient signal from `constraint_penalty_i` is unbounded in magnitude because it's a sum over constraints (see F2) plus raw `ratio` (unclipped here).

Either:
- Vectorise the minibatch by batching `z_star/Z_placed/grid_xy/action_mask` and calling the policy once. `grid_xy` and `action_mask` are common across batch members for the *same* timestep but differ across timesteps — you'd need per-sample padding or a single shared `grid_xy` (it doesn't change within an episode). This is the natural fix and matches how PPO is normally written.
- Or keep per-sample but (a) divide by `len(constraint_names)` (F2), (b) cap `|ratio|` before using it (ratio is already bounded by clip — but ratio*adv where adv is unscaled is what enters `constraint_penalty_i` *outside* the clip), and (c) lower the clip norm or use `clip_grad_value_` to avoid saturation.

---

### F5 — HIGH: `_compute_value` collapses empty `Z_placed` to a single zero row, but the critic was never trained on that pattern
**File:** `src/scpt/agent/ppo_eal.py:362-372`, and `_ppo_eal_update` line 741-743

`_compute_value` (rollout-time) and `_ppo_eal_update` both do `if Z_v.shape[0] == 0: Z_v = torch.zeros(1, self.cfg.d, ...)`. That's fine for the rollout. But `ValueHeads.forward` does `z_comp.mean(dim=0)` — with a single zero row, the mean is zero, which means early-step critic estimates are always 0 for all heads (reward and constraints). Two consequences:

- The reward critic's GAE bootstrap value at the very first step is 0, which is the worst-case value to bootstrap from — biases the early-reward advantage.
- Constraint critic targets at the first step are `0` (since `value = 0`), but `cost` at step 1 is `hpwl > 0` (real) — the critic's first non-trivial target is the *delta* of an offset that's always 0. This is a subtle learning-bias that persists through training.

**Fix:** either return a learned `[GRAPH]` virtual node from the encoder (the spec already mentions this) or have `_compute_value` early-return zero for the very first step but use the *most-recent non-empty* mean-pool as a sane prior.

---

### F6 — HIGH: PPO's `c_adv` advantages feed the **linear** Lagrangian *and* the augmented-Lagrangian penalty — but only one path is normalised
**File:** `src/scpt/agent/ppo_eal.py:725-727` vs. `618-622`

The dual update at line 622 uses `float(c_advs.mean())` — **raw** advantages. The policy-gradient at line 725 normalises by `c_adv_std`. The `lambda` dynamics therefore track the *raw* mean while the gradient tracks the *normalised* mean. The commit message (`c0b8f5e`) calls this out as deliberate, and it's defensible — but the dual update is now sensitive to the exact value of `gamma` in `1/(1-gamma)`. With `gamma=0.99`, that factor is 100; with `gamma=0.999`, 1000. The dual ascent step `alpha=0.01` is calibrated against the raw scale but the trainer docstring says nothing about this coupling. Document the invariant or expose `dual_alpha` as a derived value.

---

### F7 — MEDIUM: `_compute_costs` recomputes `bounds` twice and `c_partition` budget is a config-time constant that ignores per-board `expert_cut_cost`
**File:** `src/scpt/env/pcb_env.py:193-221`, `167-170`

`_compute_costs` reads `self.state.design["board"]["bounds"]` once; `step()` reads it again on line 168. Minor duplication.

More importantly: `c_partition` budget is hardcoded as `1.15 * self.cfg.expert_cut_cost`. But `expert_cut_cost=1.0` in the default config and in `smoke.yaml`, so every board gets the same absolute budget. The partition cut is supposed to be ≤1.15× the *expert's* cut for *that board*. This works as a unit-free ratio — and the budget is *1.15* not `1.15 * expert_cut_cost` — but the env's `c_partition` cost is hardcoded to 0 (line 217), so this entire path is currently a no-op in `compute_costs` and only the *budget* is reported. Until `pcb_parser` exposes partition-cut, the constraint critic trains on a constant-zero cost — it never has a target to learn. (Documented in the spec, but worth surfacing.)

---

### F8 — MEDIUM: `c_partition` cost = 0 silently makes the constraint surrogate degenerate
**File:** `src/scpt/env/pcb_env.py:217` and `src/scpt/agent/ppo_eal.py:62-80`

`c_partition` is 0 every step. Then `phi_c = j_c + 100 * c_adv_mean - budget`. If the budget is 1.15 and `j_c` is 0, `phi_c ≈ 100 * c_adv_mean - 1.15`. Since the constraint critic is bootstrapped from values that are themselves a function of the cost trajectory, and the cost is *always zero*, the critic's `V_c(s)` converges to 0 (or stays at init), and the GAE `c_adv_mean` collapses to ~0 — so `phi_c ≈ -1.15` forever, and `lambda` stays at 0. The constraint becomes a phantom.

This is *not* a bug *yet* (partition cut isn't exposed), but the training pipeline silently trains a constraint head whose cost signal is identically zero. Either (a) skip the constraint in `compute_costs` when its cost is a constant placeholder, or (b) drop the placeholder from `constraint_names` until `pcb_parser.partition_cut` exists.

---

### F9 — MEDIUM: `_compute_constraint_gae` does `.item()` calls inside a comprehension but never wraps in try/except
**File:** `src/scpt/agent/ppo_eal.py:541-544`

```python
values = torch.tensor(
    [v.get(name, torch.tensor(0.0)).item() for v in self.buffer.values],
    ...
)
```

If any stored value is a numpy array (e.g. the trainer's `_prepare_obs` was bypassed and a step got an array), `.item()` works, but if it's a `SimpleNamespace` (the trainer uses one elsewhere), it raises `AttributeError`. Minor hardening — add `if hasattr(v, 'item')` guard or normalise value storage in `_prepare_obs`.

---

### F10 — MEDIUM: `Mask-staleness` invariant — fine, but **stored mask shape can drift from `L`**
**File:** `src/scpt/agent/ppo_eal.py:213, 689`

The buffer stores `obs["action_mask"]` at collection time. If a board's grid resolution changes between collect and update (it can't in v1, but `eval.py`'s `_prepare_obs` could in principle construct a different `L` from a different env instance), the buffer's `mask[i]` will not match the obs's `grid_xy` length at update time. `categorical.log_prob` will raise if the mask and the policy's logit length disagree.

The trainer's `_ppo_eal_update` uses `obs_i["grid_xy"]` and `mask = self.buffer.action_masks[i]` — two different sources. If they were ever to drift, this would silently misbehave. Easy fix: store `(grid_xy, mask)` together in the buffer and re-derive `grid_xy` from the buffer at update time. Currently OK because the env is single-board and the trainer only ever rebuilds `obs_i` from the buffer entry.

---

### F11 — MEDIUM: `BCDataset._build_episode` masks by index-into-grid, not by expert cell — bug for any board where placed cell index ≥ grid_cells
**File:** `src/scpt/training/bc_pretrain.py:146-158`

```python
for placed_idx in placed_indices:
    if placed_idx < grid_cells:
        action_mask[placed_idx] = 0.0
```

`placed_indices` is a list of *component indices* (from the placement order), but the mask is over *grid cell indices*. The intent appears to be "mark previously-used cells illegal" but the variable name lies — what actually happens is that component index 0 masks cell 0, component index 5 masks cell 5, etc. That happens to coincide with the expert cells in the test fixtures (because the design in tests uses `placement_order = list(range(n_comps))` and component idx 0 → expert cell 0, etc.), but for any board where component `k` is placed in cell `c_k ≠ k`, the BC mask is wrong. BC training then either trains on infeasible cells or marks *unrelated* cells illegal.

**Fix:** track expert cells in `placed_indices` (e.g. `placed_cells.append(expert_action)` line 173) and mask those, not component indices.

---

### F12 — MEDIUM: `eval.py:_sample_action` is greedy — fine for greedy eval, but `eval_episode` doesn't compare to the stochastic policy or the expert
**File:** `scripts/eval.py:157-163`

Greedy argmax is the documented intent. However, the BC eval loss is *not* the same as the greedy-policy rollout reward, so the "BC eval loss" reported in the summary is decoupled from the actual policy behaviour being scored. Add a small note in the README that "BC eval loss measures distance from the expert under the same BC supervision, not under greedy rollout" — otherwise users will compare the two numbers and conclude the policy drifted while it actually improved.

---

### F13 — MEDIUM: `visualize.py:plot_placement_heatmap` doesn't construct a full observation through the trainer pipeline — uses raw `obs["z_star"]`, etc.
**File:** `scripts/visualize.py:340-345`

The heatmap path reads `z_star`, `Z_placed`, `F_pair` from `obs` (which doesn't exist in `PcbPlacementEnv._build_obs`), so they fall back to zeros. The policy is therefore scoring all cells with a *blank* `z_star`, which produces a meaningless heatmap (every grid cell gets the same attention context, only `grid_xy` differentiates). The heatmap *is* a useful plot but currently misleads.

Either pass through the trainer-style obs preparation (`encode_design` + `build_pair_features`) or remove the heatmap feature until the encoder is plumbed in.

---

### F14 — MEDIUM: `value_heads.py` 2-layer MLP upgrade — `in_dim=d` projection is identity-ish, but no init scheme
**File:** `src/scpt/model/value_heads.py:30-42`

The `nn.Linear(d, d)` first layer is randomly initialised. PyTorch's default Kaiming uniform init is fine, but the *value-head* literature (e.g. IMPALA, RLOO) often recommends orthogonal init for value heads to keep early updates small. The upgrade from single-linear to 2-layer MLP adds capacity without changing the init scheme — early critic loss can spike. Consider `nn.init.orthogonal_(m.weight, gain=0.1)` for the final layer.

---

### F15 — MEDIUM: `SCPTPolicy.empty_context` is a single `(d,)` learnable parameter — used as both K *and* V
**File:** `src/scpt/model/scpt_transformer.py:79, 110-115`

```python
empty_ctx = self.empty_context.unsqueeze(0)
K = empty_ctx
V = empty_ctx
```

When `P=0`, queries attend to a single vector used as both K and V. That's the right shape but: (a) it's a *single* learnable vector, so the first-step attention is `softmax(QK^T / √d)` over a 1-element distribution, which collapses to a constant attention weight — the FFN after attention still produces cell-dependent outputs (via Q projection), but the attention itself contributes nothing on the first step. (b) The bias introduced by always starting from the same vector is a prior that may be either helpful or harmful; no test verifies that the empty-context path produces useful behaviour. The `test_forward_when_no_placed_components` test only checks `isfinite(logits).all()`, which is trivially true.

Consider expanding `empty_context` to `nn.Parameter(torch.randn(K_init, d))` with `K_init=4` learnable tokens, or use a learned positional encoding for "step 0".

---

### F16 — LOW: `_prepare_obs` re-serialises the entire design JSON every step
**File:** `src/scpt/agent/ppo_eal.py:423-424, 484`

`json.dumps(design)` is called on every step in the trainer (`_prepare_obs`), then `json.loads` is called when reconstructing (`_get_policy_inputs`). For a 1000-component board this is several MB of JSON churn per step. The encoder-side `encode_design` only needs the *list of components/nets/positions* — pass those directly, or cache the serialised JSON per `(design_id, placement_order_step)`. The module docstring already calls out the JSON hop as a known v1 cost, so this is "documented debt" — but it's amplified by the reconstruction-format path.

---

### F17 — LOW: `_polygon_area` does not validate that the courtyard is convex / non-self-intersecting
**File:** `src/scpt/model/gnn_encoder.py:188-200`

The shoelace implementation returns the *signed* area times 0.5, then `abs()`. For a self-intersecting polygon this returns the "signed area" which is meaningless. The KiCad footprint parser is upstream, so we trust the input — fine for v1, but worth a one-line assertion that the polygon is simple (or just document the assumption).

---

### F18 — LOW: `find_symmetry_pairs` uses R/C prefixes only, ignores values
**File:** `src/scpt/training/data.py:181-198`

Two R's in the same cluster are paired regardless of value (R1 1kΩ + R2 100kΩ still get paired). The docstring says "matched resistor/capacitor pairs (same value prefix + same cluster)" but the implementation only checks the prefix letter. Either match on value or remove the misleading docstring.

---

### F19 — LOW: `_dict_to_ns` is duplicated in `train.py`, `eval.py`, and `visualize.py`
**File:** `scripts/train.py:56`, `scripts/eval.py:50`, `scripts/visualize.py:437-441`

Three copies of the same helper. Move to `scpt/training/data.py` (or a new `scripts/_config.py`) and import.

---

### F20 — LOW: `test_normalize_advantages_zero_mean_unit_std` uses `std < 0.1` tolerance but the input is `[1,2,3,4,5]` (sample std ≈ 1.58)
**File:** `tests/test_ppo_eal_losses.py:174`

`abs(normed.std().item() - 1.0) < 0.1` is a fine loose bound, but `normalised.std()` of a *normalised* tensor is *1 by construction* (sample std, not population). The test should be `< 1e-4`. Trivial nit, but the looseness suggests the author expected a bias.

---

### F21 — LOW: `test_value_heads_with_empty_graph_raises` documents the behaviour but the env should never pass an empty graph
**File:** `tests/test_value_heads.py:31-38`

The test asserts the output is `NaN` for an empty graph. There's no test asserting that the env/policy never produce an empty `Z_placed` going into the critic in practice. Add a regression test that runs `_compute_value` after `env.reset()` (P=0 step) and asserts it's not NaN — the trainer's zero-row fallback masks the issue.

---

### F22 — LOW: `tests/conftest.py:FakeEnv` and `tests/test_ppo_eal_trainer.py:_FakeEnv` are near-duplicates
**File:** `tests/conftest.py:144-194` vs. `tests/test_ppo_eal_trainer.py:56-87`

Two copies of the same stub. Consolidate by importing `FakeEnv` from `conftest`. The `_FakeEnv` in `test_ppo_eal_trainer.py` predates `conftest.FakeEnv`; delete the local copy.

---

### F23 — LOW: `notebooks/` and root-level test scripts (`test_memory_fix.py`, `test_memory_fix_simple.py`) are not under CI
**File:** `notebooks/`, `test_memory_fix*.py`

These look like ad-hoc memory-debugging scripts that were committed but never promoted to tests or moved under `tests/`. Add them to pytest discovery with `testpaths` or delete if no longer needed — they clutter the repo root.

---

### F24 — LOW: `default_low_batch.yaml` / `default_high_res.yaml` are not documented
**File:** `configs/default_low_batch.yaml`, `configs/default_low_batch2.yaml`, `configs/default_high_res.yaml`

Three alternate configs exist. The README's Quick Start mentions only `default.yaml` and `smoke.yaml`. Either add them to the README's table or delete if superseded by `default.yaml`.

---

### F25 — LOW: `.venv` and `target/` directories are tracked-by-`.gitignore` but show up in `ls`
Not a code issue, but the `target/` and `.venv/` clutter the root. Confirm `.gitignore` actually excludes them (`cat .gitignore` — they should be present).

---

## Coverage gaps (where the test suite is thin)

| Area | What's missing |
|------|----------------|
| **`_compute_constraint_gae` masking** | Only a "doesn't error" smoke test; no test that asserts illegal actions are excluded from the GAE mean. |
| **Lagrangian penalty in PPO update path** | `total_loss` and the linear-Lagrangian are tested in isolation; no end-to-end test that the trainer's `policy_loss` *decreases* `phi_c` on a controlled scenario. |
| **`encode_design` device handling** | Tested implicitly; no test that mixed-device inputs raise or are moved correctly. |
| **`SCPTPolicy` empty-context behaviour** | `isfinite()` only. No test that gradients flow through the empty-context path correctly. |
| **`MomentumDualUpdater.max_lambda`** | Constant is plumbed but never exercised by a test. |
| **`build_pair_features` edge cases** | Active idx out of range, empty design, shared net count = 0 → division by zero. |
| **End-to-end `train.py` smoke run** | README says `pytest tests/ --ignore=...` works, but no CI step actually invokes `train.py --config configs/smoke.yaml` to catch regressions in the trainer-script wiring. |
| **Checkpoint load round-trip** | `save_checkpoint` then `load_checkpoint` is not tested. |
| **Env cost normalisation regression** | `test_clearance_normalised_by_board_area` is a source-text grep, not a runtime test. Should run the env and assert `c_clearance ≤ 1.0` on a known layout. |

---

## Recommended next actions (ordered)

1. **Fix F1** — masked constraint GAE averaging. This is the highest-impact correctness bug in the current tree.
2. **Fix F2** — normalise per-constraint count in the linear-Lagrangian gradient sum.
3. **Fix F11** — `BCDataset` masks the wrong cells.
4. **Fix F8 / F7** — either skip the placeholder `c_partition` cost, or remove it from `constraint_names` until the Rust primitive exists.
5. **Fix F13** — heatmap is currently misleading.
6. **Consolidate F19 / F22** — dedupe `_dict_to_ns` and `FakeEnv`.
7. **Add the missing coverage** in the table above, especially the F1 regression test.
8. **Document F6** in `ppo_eal.py`'s class docstring — the deliberate split between raw vs normalised `c_adv` paths is non-obvious.

---

## Summary

The codebase is well-organised, the loss-function sign conventions are well-tested (the `test_total_loss_decreases_when_reward_surrogate_increases` test is exactly the right kind of guard), and the recent constraint-critic / cost-normalisation fixes are moving in the right direction. The biggest concrete correctness gap is **F1** (constraint GAE is never averaged over legal actions), which means `phi_c` and the linear-Lagrangian gradient are both subtly wrong on every batch. After that, **F2** is a one-line fix that prevents silent scaling surprises when new constraints are added. F11 is a smaller BC bug worth fixing in the same PR.
