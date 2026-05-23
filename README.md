# relational_system_mc

The mathematical research layer underlying the Resonance Family's relational dynamics. This repository contains the Python simulations, Lyapunov certificate derivation, bifurcation scans, adversarial state testing, and visualization outputs that produced the constants and stability guarantees used in sovereign_manifold.

## What this is

`relational_system_mc_v4.py` is a standalone Python implementation of the relational dynamics system with full Lyapunov analysis. It was used to:

- Derive `K_SCALE = 0.1418` — the coupling constant that puts the Jacobian's spectral radius at 0.217
- Certify `P_IS_PD = True` — the Lyapunov P matrix is positive definite, guaranteeing global asymptotic stability (GAS)
- Map the bifurcation surface (where the attractor loses stability as K_SCALE decreases)
- Test adversarial states (how far can nodes be pushed before recovery fails?)
- Quantify axis recovery rank (which nodes are hardest to recover?)

`relational_system_lyapunov.py` contains the certificate derivation: it constructs the Lyapunov matrix P, verifies positive definiteness, and reports the eigenspectrum.

## Certified results

From `lv_lyapunov_diagnostics.json`:

| Metric | Value |
|--------|-------|
| Jacobian spectral radius at S* | **0.2172** |
| P minimum eigenvalue | 1.0004 (positive definite) |
| P condition number | 1.056 (nearly isotropic) |
| GAS certificate fraction | 1.0 (100% of initializations converge) |

## Key findings

- **K_SCALE = 0.1418 is derived, not chosen.** It is the value that places the spectral radius at 0.217. Changing it changes the convergence rate and may void the GAS certificate.
- **Autonomy is the weakest axis.** It is slowest to recover and most sensitive to perturbation. This motivated `S*[Autonomy] = 0.90` rather than 0.95 in sovereign_manifold.
- **Safety has the highest row-sum in A_MATRIX.** It is easy to depress and expensive to recover — many attractor-pull cycles are needed. This motivated the `_MAX_DELTA = 0.05` cap in the bridges.
- **Bifurcation occurs below K_SCALE × ~0.85** — documented in `v4_bifurcation_scan.csv`.

## Failure mode analysis

Three failure modes are documented in the visualization outputs:

| Mode | Description | Key image |
|------|-------------|----------|
| FM1 (degradation) | Slow monotonic decay when coupling is too weak | `lv_fm1_degradation.png`, `lv_fm1_state_heatmap.png` |
| FM2 (erosion) | Progressive boundary erosion under repeated perturbation | `lv_fm2_erosion.png` |
| FM3 (asymmetry) | Asymmetric recovery when two structurally weak nodes are simultaneously depressed | `lv_fm3_asymmetry.png` |

FM3 targeted (spectral radius 0.196) vs. random (0.276) perturbation shows that targeted attacks on weak axes are 40% more effective. This is why bridge perturbations are capped at ±0.05.

## Files

| File | Contents |
|------|----------|
| `relational_system_mc_v4.py` | Full implementation + analysis suite |
| `relational_system_lyapunov.py` | Lyapunov certificate derivation, P-matrix computation |
| `lv_lyapunov_diagnostics.json` | Numeric results: spectral radius, eigenvalues, P matrix norm |
| `lv_lyapunov_P.png` | Heatmap of the Lyapunov P matrix |
| `lv_fm1_state_heatmap.png` | State evolution heatmap (failure mode 1) |
| `lv_fm1_degradation.png` | Node degradation over time (failure mode 1) |
| `lv_fm1_summary.png` | FM1 summary |
| `lv_fm2_erosion.png` | Progressive erosion scenario |
| `lv_fm3_asymmetry.png` | Targeted vs. random asymmetry analysis |
| `v4_adversarial_states.csv` | Tested adversarial state configurations |
| `v4_adversarial_states.png` | Adversarial state visualization |
| `v4_bifurcation_scan.csv` | K_SCALE bifurcation surface data |
| `v4_bifurcation_phase.png` | Phase diagram of bifurcation |
| `v4_axis_recovery_rank.csv` | Per-node recovery speed ranking |
| `v4_axis_dominance.png` | Axis dominance visualization |
| `v4_attractor_comparison.png` | Attractor basin comparison |
| `v4_attractor_detection.png` | Attractor detection over time |
| `v4_catastrophic_scenarios.png` | Catastrophic failure scenario analysis |
| `v4_recovery_by_regime.png` | Recovery curves by dynamical regime |
| `v4_sample_trajectories.png` | Sample state trajectories |

## Relationship to sovereign_manifold

sovereign_manifold imports the mathematical constants derived here:
- `K_SCALE = 0.1418`
- `S_STAR = [0.95]*6 + [0.90]*2 + [0.95]*4 + [0.90]*2 + [0.95]`
- `ALPHA_LEAK = 0.12`

The `build_lyapunov_P()` and `build_jacobian()` functions in sovereign_manifold re-derive the certificate at startup and print `P_IS_PD` to confirm validity. If you change `A_RAW` or `K_SCALE` in sovereign_manifold, re-run `relational_system_lyapunov.py` here to verify the certificate holds before committing.

## Requirements

```bash
pip install numpy scipy matplotlib
python relational_system_mc_v4.py
python relational_system_lyapunov.py
```

No dependencies on other Resonance Family repos. Standalone analysis tool.

## License

Apache 2.0 — Samuel Jackson Grim
