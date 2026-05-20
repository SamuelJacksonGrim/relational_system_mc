# CLAUDE.md — relational_system_mc

This is the canonical source for the relational dynamics math. Other repos copy or depend on the constants defined here. Changes here have cascading consequences.

## Canonical source for A_MATRIX, S_STAR, K_SCALE

These three quantities are the load-bearing foundation of the architecture:

- `A_MATRIX`: the 15×15 coupling matrix (`A_RAW × K_SCALE`)
- `S_STAR`: the target attractor — `[0.95]*6 + [0.90]*2 + [0.95]*4 + [0.90]*2 + [0.95]`
- `K_SCALE = 0.1418`: derived from spectral radius target, not chosen arbitrarily

`sovereign_manifold.py` contains inline copies of all three. If you modify them here, update `sovereign_manifold.py` manually — there is no automatic sync.

## K_SCALE is derived, not arbitrary

K_SCALE = 0.1418 was computed so the Jacobian at S* has spectral radius 0.217, making the linearized system contractive. This is the formal GAS (global asymptotic stability) guarantee. Do not change K_SCALE without re-running the Lyapunov analysis and verifying P is positive definite.

## ALPHA_LEAK = 0.12 — leak term, not a learning rate

The leak term subtracts `alpha × s` from the input sum. It is not a decay rate toward zero and not a learning rate. It modulates how much of the current state "bleeds through" unprocessed. Changing it shifts the effective attractor location and invalidates the existing Lyapunov certificate.

## S* is not uniform 0.95

Nodes 6 (Boundaries), 7 (Autonomy), 12 (Learning), 13 (Adaptability) target 0.90. This reflects A_MATRIX structure: Autonomy(7) has the smallest stability margin (FM3 fragility point), and Growth nodes (12, 13) are intentionally less tightly constrained than structural nodes.

## Autonomy(7) is the Lyapunov fragility point

By formal Lyapunov mode analysis (FM3), Autonomy(7) has the smallest stability margin and is the first node to detach from S* under perturbation. The bridge perturbation cap of ±0.05 is calibrated so that worst-case perturbation on Autonomy(7) stays within the basin.

## Safety(14) has the highest A_MATRIX connectivity

The Safety(14) row sum in A_RAW is the highest across all nodes. This means Safety is heavily supported by all other nodes, but also that external downward pressure on Safety(14) is resisted strongly — it is expensive to suppress and expensive to recover.

## Lyapunov check before any structural update

Before committing any change to `A_RAW`, `K_SCALE`, `ALPHA_LEAK`, or `S_STAR`:

```python
J = build_jacobian(A_MATRIX, S_STAR, B_NOMINAL)
P, eigvals, is_pd = build_lyapunov_P(J)
assert is_pd, "Lyapunov certificate lost — change rejected"
```

This is a hard requirement, not a recommendation. A change that passes tests but fails this check is rejected.
