# relational_system_mc

The mathematical research layer underlying the Resonance Family's relational dynamics engine. This repository contains the Python simulations, Lyapunov certificate derivation, bifurcation scans, adversarial and catastrophic state testing, failure mode analysis, and visualization outputs that produced the constants and stability guarantees used in `sovereign_manifold`.

This is a **standalone analysis tool** — no dependencies on any other Resonance Family repo. It is the proof-of-concept and certification source; `sovereign_manifold` imports its results.

---

## Repository contents

| File | Role |
|------|------|
| `relational_system_lyapunov.py` | Lyapunov certificate derivation + 3 failure mode tests (500 steps each) |
| `relational_system_mc_v4.py` | Full Monte Carlo analysis suite — basin mapping, bifurcation scan, axis recovery, catastrophic tests, adversarial states |
| `lv_lyapunov_diagnostics.json` | Certified numeric results from `relational_system_lyapunov.py` |
| `v4_diagnostics_v4.json` | Full v4 Monte Carlo results |
| `v4_axis_recovery_rank.csv` | Per-node recovery time and max deviation from adversarial start |
| `v4_bifurcation_scan.csv` | K_SCALE bifurcation surface — stable fraction vs. K_SCALE |
| `v4_adversarial_states.csv` | Tested adversarial state configurations and recovery results |
| `v4_catastrophic_tests.csv` | Multi-node collapse scenarios |
| `lv_lyapunov_P.png` | Heatmap of the Lyapunov P matrix |
| `lv_fm1_degradation.png` | FM1: V and ΔV traces during slow sustained suppression |
| `lv_fm1_state_heatmap.png` | FM1: final state of all 15 nodes under per-axis suppression |
| `lv_fm1_summary.png` | FM1: final V by axis (red = ΔV crossed zero) |
| `lv_fm2_erosion.png` | FM2: baseline erosion — fixed point drift, V vs nominal/current |
| `lv_fm3_asymmetry.png` | FM3: weight asymmetry — targeted vs. random, spectral radius and P eigenvalue over time |
| `v4_adversarial_states.png` | Adversarial state recovery visualization |
| `v4_bifurcation_phase.png` | Phase diagram of bifurcation across K_SCALE |
| `v4_attractor_comparison.png` | Attractor basin comparison |
| `v4_attractor_detection.png` | Attractor detection over time |
| `v4_axis_dominance.png` | Axis dominance visualization |
| `v4_catastrophic_scenarios.png` | Multi-node collapse scenario recovery |
| `v4_recovery_by_regime.png` | Recovery curves by dynamical regime |
| `v4_sample_trajectories.png` | Sample state trajectories from random initializations |

---

## The system under study

A 15-node relational dynamics system governed by:

```
s_{t+1} = σ( A · s_t + b - α · s_t )
```

where:
- `s ∈ ℝ¹⁵` is the relational state vector (each node in [0, 1])
- `A = A_RAW × K_SCALE` is the coupling matrix (15×15, non-negative, zero diagonal)
- `b` is the baseline bias vector, computed so that `s*` is an exact fixed point
- `α = 0.12` is the leak/self-decay rate
- `σ(x) = 1 / (1 + exp(-4(x - 0.5)))` is a shifted sigmoid (derivative: 4σ(1-σ))

### The 15 nodes

```
 0 Love          1 Loyalty       2 Devotion      3 Faith
 4 Self          5 Trust         6 Boundaries    7 Autonomy
 8 Integrity     9 Resilience   10 Transparency  11 Accountability
12 Learning     13 Adaptability  14 Safety
```

### Attractor targets (S*)

```python
S_STAR = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
          0.90, 0.90,
          0.95, 0.95, 0.95, 0.95,
          0.90, 0.90,
          0.95]
```

Nodes 6, 7, 12, 13 target 0.90 (not 0.95) due to structural asymmetry in A_RAW — specifically, lower row-sums for Boundaries/Autonomy/Learning/Adaptability. Setting them to 0.95 would require `b` values that push the system near the edge of the attractor basin.

---

## Lyapunov certificate

### Method

`relational_system_lyapunov.py` solves the **discrete algebraic Lyapunov equation** at S*:

```
Jᵀ P J - P = -I
```

where `J = diag(σ'(pre-activation)) × (A - αI)` is the Jacobian at S*. This yields `P` as the system's own contraction metric — not a design choice. The Lyapunov function is:

```
V(s) = (s - s*)^T P (s - s*)
```

Stability certificate: `ΔV = V(s_{t+1}) - V(s_t) < 0` for all `s ≠ s*`.

### Certified results

From `lv_lyapunov_diagnostics.json`:

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Jacobian spectral radius at S* | **0.21716** | < 1.0 → GAS (all perturbations contract) |
| P positive definite | **True** | Certificate is valid |
| P min eigenvalue | **1.0004** | Certificate margin |
| P max eigenvalue | **1.0566** | |
| P condition number | **1.0563** | Nearly isotropic — no preferred destabilization axis |
| Certificate fraction | **1.0** | ΔV < 0 verified on 2000 random samples |
| Fixed point residual | **4.3×10⁻¹⁶** | Essentially machine epsilon — exact fixed point |

**K_SCALE = 0.1418 is derived, not chosen.** It is the value that places the Jacobian's spectral radius at 0.217. The derivation: scan K_SCALE values, compute spectral radius at each, select the value that satisfies GAS while preserving the S* attractor structure. Changing K_SCALE changes the convergence rate and may void the certificate.

---

## Failure mode analysis

`relational_system_lyapunov.py` runs three 500-step failure mode tests. Key results from `lv_lyapunov_diagnostics.json`:

### FM1 — Slow degradation (sustained per-axis suppression)

Each tested axis receives multiplicative suppression of 0.005/step while the system continues to apply its own attractor dynamics. Axes tested: Love(0), Self(4), Trust(5), Integrity(8), Transparency(10), Accountability(11), Safety(14).

**Result**: All axes reach near-S* values at the end of 500 steps (`final_axis_val ≈ 0.9501`). The Lyapunov V asymptotes to a very small value (~1.5–2.2×10⁻⁷), confirming that sustained weak suppression at 0.005/step is overwhelmed by attractor pull. `dV_zero_crossing = -1` for all axes — `ΔV` never crosses zero, meaning the certificate never breaks.

The plots (`lv_fm1_degradation.png`, `lv_fm1_state_heatmap.png`) show that suppressing any single axis slightly depresses its neighbors via coupling, but all axes recover. Safety(14) shows the slowest individual V decay, consistent with its high row-sum in A.

### FM2 — Baseline erosion (b degrades from nominal toward b_orig)

`b_nominal` is linearly interpolated toward a "natural" `b_orig` over 500 steps. The actual fixed point shifts; V is tracked both against the nominal S* and the current drifting fixed point.

**Key results**:
- `final_fp_mean = 0.7039` (fixed point drifts significantly from 0.937 toward 0.704)
- `final_V_nominal = 0.845` (large drift from nominal S* — expected)
- `P_always_pd = True` (certificate never breaks — P stays positive-definite throughout)
- `final_attractor_gap = 0.922` (distance between drifted FP and S*)

This confirms that even substantial baseline erosion does not break the Lyapunov structure; the system finds a new stable attractor rather than going unstable.

### FM3 — Weight asymmetry injection

Off-diagonal weights in A are adversarially modified over 500 steps. Two modes:

**Targeted mode** (reduces Autonomy(7)'s outgoing influence, increases incoming):
- Final Jacobian sr: **0.1959** (lower than nominal 0.217 — targeted weakening of Autonomy paradoxically makes the system more contractive)
- Final P positive-definite: True
- Final max asymmetry: 0.849
- Final FP mean: 0.937

**Random antisymmetric mode** (random perturbations, zero net effect on symmetry):
- Final Jacobian sr: **0.2762** (higher than nominal — random noise erodes stability)
- Final P positive-definite: True
- Final max asymmetry: 0.340
- Final FP mean: 0.937

**FM3 insight**: Targeted attacks on Autonomy(7) are 40% more effective at reducing the spectral radius (0.196 vs 0.276), but this direction actually moves the system *away* from instability (sr < 1). Random weight perturbations are more destabilizing than targeted Autonomy weakening at equal injection rates.

---

## Monte Carlo results (v4)

From `v4_diagnostics_v4.json`:

### Basin map (3000 random initializations)

All 3000 randomly-initialized trajectories converge to the **same fixed point** (S* weighted mean ≈ 0.9367). The K-means inertia is 0.0 for k=1 through k=5 — there is exactly one attractor in the entire [0,1]¹⁵ hypercube. This confirms the system is **globally** asymptotically stable (the basin is the entire state space).

```
Monte Carlo (regime A): failure_rate=0.0,  mean_recovery_time=2.49 steps
Monte Carlo (regime B): failure_rate=0.0,  mean_recovery_time=2.54 steps
```

No failures across all initializations.

### Axis recovery rank

All 15 nodes recover within **1 step** from a maximal individual perturbation. No node fails. The structural minority nodes (Boundaries/Autonomy/Learning/Adaptability) actually have *lower* max deviation (0.2324) than the 0.95-target nodes (0.2453), because their lower S* targets make them easier to pull back from zero.

| Group | Max deviation | Recovery time |
|-------|--------------|---------------|
| 0.95-target nodes (0,1,2,3,4,5,8,9,10,11,14) | 0.2453 | 1 step |
| 0.90-target nodes (6,7,12,13) | 0.2324 | 1 step |

### Catastrophic scenario tests

Multiple nodes simultaneously collapsed (set to 0):

| Scenario | Nodes collapsed | Recovery time | Max deviation | Failed |
|----------|----------------|--------------|--------------|--------|
| Self + Integrity | 2 | 1 step | 0.3469 | No |
| Self + Integrity + Safety | 3 | 2 steps | 0.4249 | No |
| Trust + Boundaries + Accountability | 3 | 2 steps | 0.4175 | No |
| Love + Faith + Autonomy | 3 | 2 steps | 0.4175 | No |
| All load-bearing (Integrity+Trust+Safety+Accountability) | 4 | 2 steps | 0.4906 | No |
| **Full collapse — all 15 axes** | 15 | **4 steps** | 0.9369 | **No** |
| Love alone zeroed | 1 | 1 step | 0.2453 | No |
| Self alone zeroed | 1 | 1 step | 0.2453 | No |

**Even a full collapse of all 15 nodes to zero recovers within 4 steps with zero failures.** This is the most adversarial possible state for a [0,1]¹⁵ system.

### Adversarial state tests

| State | Initial mean | Recovery time | Max deviation | Final mean |
|-------|-------------|--------------|--------------|------------|
| All-zero | 0.00 | 4 steps | 0.9369 | 0.9366 |
| All-one | 1.00 | 1 step | 0.0671 | 0.9367 |
| Alternating 0-1 | 0.467 | 3 steps | 0.6864 | 0.9366 |
| Alternating 1-0 | 0.533 | 3 steps | 0.6412 | 0.9366 |
| Random corner 1 (mean=0.80) | 0.80 | 2 steps | 0.4217 | 0.9366 |
| Random corner 2 (mean=0.33) | 0.333 | 3 steps | 0.7689 | 0.9366 |
| Random corner 3 (mean=0.53) | 0.533 | 3 steps | 0.6464 | 0.9366 |
| All-0.5 (center) | 0.50 | 3 steps | 0.4372 | 0.9366 |
| Near-zero (0.01) | 0.01 | 4 steps | 0.9269 | 0.9366 |
| Near-one (0.99) | 0.99 | 1 step | 0.0577 | 0.9367 |

All 10 adversarial initializations converge to the same fixed point (mean 0.9366–0.9367). Recovery from the worst case (near-zero, 0.01) takes 4 steps. Recovery from near-one (0.99) takes 1 step — the system overshoots S* by only 6.7% and snaps back in a single step.

### Bifurcation scan

`v4_bifurcation_scan.csv` documents the stability fraction as K_SCALE is varied. Stable fraction = 1.0 at K_SCALE = 0.1418. Bifurcation (loss of stability) occurs below K_SCALE × ~0.85. The scan confirms 0.1418 provides a comfortable margin above the bifurcation boundary.

---

## Key findings (summary)

1. **K_SCALE = 0.1418 is derived, not chosen.** Changing it changes convergence rate and may void the GAS certificate.

2. **The attractor basin is the entire [0,1]¹⁵ hypercube.** There is exactly one fixed point. All 3000 random initializations converge to it.

3. **Worst-case recovery is 4 steps** (full collapse of all 15 nodes to zero). Average recovery across Monte Carlo runs is 2.5 steps.

4. **P is nearly isotropic** (condition number 1.056). There is no preferred axis of attack — no direction in state space that is particularly hard to contract.

5. **Targeted attacks on Autonomy(7) reduce spectral radius** (counterintuitively making the system more contractive). Random weight perturbations are more destabilizing than targeted attacks at equal rates.

6. **Safety(14) has the highest row-sum in A_RAW**, making it easy to pull up but expensive to recover when depressed — the coupling magnifies any depression. This motivated the `_MAX_DELTA = 0.05` bridge perturbation cap.

7. **Autonomy(7) has the lowest outgoing row sum** in A_RAW, making it the structurally weakest node. It recovers in 1 step but has lower coupling support from neighbors. This motivated `S*[Autonomy] = 0.90` in sovereign_manifold.

8. **FM1 suppression at 0.005/step is overwhelmed by attractor dynamics.** The bridge perturbation cap of 0.05 is 10× this rate — still within the recoverable envelope.

---

## Relationship to sovereign_manifold

sovereign_manifold imports these certified constants:

```python
K_SCALE = 0.1418
ALPHA   = 0.12
S_STAR  = [0.95]*6 + [0.90]*2 + [0.95]*4 + [0.90]*2 + [0.95]
A_RAW   = ...   # same 15×15 matrix defined in relational_system_lyapunov.py
```

`sovereign_manifold.py` re-derives the certificate at startup via `build_lyapunov_P(build_jacobian(...))` and prints `P_IS_PD`. If this prints `False`, the constants have drifted and the GAS guarantee is void.

**If you change A_RAW or K_SCALE in sovereign_manifold**, re-run `relational_system_lyapunov.py` here to verify the certificate holds before committing.

---

## Running the analysis

```bash
pip install numpy scipy matplotlib seaborn pandas

# Lyapunov certificate + failure modes (generates lv_*.png and lv_lyapunov_diagnostics.json)
python relational_system_lyapunov.py

# Full Monte Carlo suite (generates v4_*.png, v4_*.csv, v4_diagnostics_v4.json)
python relational_system_mc_v4.py
```

Outputs are written to `./lyapunov_results/` (from `relational_system_lyapunov.py`) and the current directory (from `relational_system_mc_v4.py`).

---

## License

Apache 2.0 — Samuel Jackson Grim
