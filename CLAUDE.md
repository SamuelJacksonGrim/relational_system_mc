# CLAUDE.md — relational_system_mc

## Role of this repository

Research and formal analysis, not production code. This is where the mathematical properties of the relational dynamics were proven, not where they run. Production execution is in sovereign_manifold.

## Why the math matters

The 15 nodes aren't arbitrary labels. Love, Trust, Autonomy, Boundaries, Safety are named relational constructs modeled as mathematical objects. The proof here establishes that a system built around those constructs has a single global attractor, recovers from total collapse in 4 steps, and contracts uniformly in all directions (P condition number 1.056). That's what makes sovereign_manifold's relational dynamics trustworthy rather than just plausible. The DRA modes (GENERATOR / OBSERVER / WATCHER) that sovereign_manifold uses at runtime are observational labels over a system that this repository proves has no stable failure states.

## The constants here are authoritative

K_SCALE, S_STAR, and ALPHA_LEAK values were derived by running this codebase's Lyapunov analysis. If sovereign_manifold's constants ever drift from the values here, this repo is correct — sovereign_manifold is out of sync.

Current certified values:
- `K_SCALE = 0.1418` → spectral radius 0.2172, GAS guaranteed
- `S_STAR[6] = S_STAR[7] = S_STAR[12] = S_STAR[13] = 0.90` (structurally weak axes)
- `P_IS_PD = True`, condition number 1.056

## Re-deriving after A_MATRIX changes

If you modify A_MATRIX (the coupling weights) in any repo:
1. Update `relational_system_mc_v4.py` with the new A_RAW values
2. Run `relational_system_lyapunov.py` — verify `P_IS_PD = True` and the new spectral radius
3. If spectral radius has changed significantly (>0.05 from 0.217), re-scan the bifurcation surface
4. Update K_SCALE in sovereign_manifold only after the certificate is confirmed

The Lyapunov P matrix should remain near-isotropic (condition number < 2.0). A condition number above 5.0 indicates that some axes have become dramatically harder to stabilize than others — that is a sign the A_MATRIX change is structurally problematic.

## Failure mode analysis — what the FM images represent

- **FM1 (degradation)** — slow monotonic decay when K_SCALE is too low. The system converges to a depressed fixed point, not the target S*. This is the sub-bifurcation regime.
- **FM2 (erosion)** — progressive erosion under repeated additive perturbation. Demonstrates why `_MAX_DELTA = 0.05` is necessary: without the cap, bridge perturbations can push the system into the FM2 regime.
- **FM3 (asymmetry)** — targeted attacks on structurally weak axes (Autonomy + Learning simultaneously) are 40% more damaging than random attacks of the same magnitude. This motivates monitoring low-S* nodes specifically.

## Adversarial states

`v4_adversarial_states.csv` documents tested configurations where one or more nodes are initialized far from S*. All tested configurations recover — this is the GAS guarantee. The recovery time varies across test regimes: Safety recovers slowest (highest row-sum in A_MATRIX), Autonomy recovers second-slowest.

## This repo has no runtime dependencies

No imports from other Resonance Family repos. Pure Python (numpy, scipy, matplotlib). It is a standalone analysis tool — adding stack dependencies would undermine its role as an independent certificate validator.
