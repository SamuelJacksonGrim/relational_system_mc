"""
relational_system_lyapunov.py

Lyapunov stability analysis + three failure mode tests for the 15-node
relational system.

LYAPUNOV CONSTRUCTION
  Solve the discrete algebraic Lyapunov equation at the operating point:
      J^T P J - P = -I
  where J = diag(sigma'(A*s* + b - alpha*s*)) @ (A - alpha*I)
  This yields P as the system's own contraction metric — not a design choice.
  Certificate: V(s) = (s-s*)^T P (s-s*)
  Stability: ΔV = V(s_{t+1}) - V(s_t) < 0 for all s ≠ s*

FAILURE MODE TESTS
  FM1 — Slow degradation:    sustained per-axis suppression over 500 steps.
                              ΔV/step is the early-warning signal.
  FM2 — Baseline erosion:    b degrades linearly toward zero.
                              Track V against both nominal and drifting s*.
  FM3 — Weight asymmetry:    off-diagonal weights modified adversarially
                              over time. Recompute P; flag when it goes
                              non-positive-definite.

Nodes:
  0 Love, 1 Loyalty, 2 Devotion, 3 Faith, 4 Self,
  5 Trust, 6 Boundaries, 7 Autonomy, 8 Integrity, 9 Resilience,
  10 Transparency, 11 Accountability, 12 Learning, 13 Adaptability, 14 Safety

Dependencies:
  pip install numpy pandas matplotlib seaborn scipy

Run:
  python relational_system_lyapunov.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve_discrete_lyapunov
import os, json, time

# ── Config ───────────────────────────────────────────────────────────────────
np.random.seed(42)

N_NODES   = 15
ALPHA     = 0.12
K_SCALE   = 0.1418
FM_STEPS  = 500       # steps per failure mode test
OUTPUT_DIR = "lyapunov_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODE_NAMES = [
    "Love", "Loyalty", "Devotion", "Faith", "Self",
    "Trust", "Boundaries", "Autonomy", "Integrity", "Resilience",
    "Transparency", "Accountability", "Learning", "Adaptability", "Safety"
]

# ── Matrix & attractor ───────────────────────────────────────────────────────
A_raw = np.array([
    [0,    0.60, 0.50, 0.40, 0.70, 0.55, 0.45, 0.30, 0.50, 0.45, 0.55, 0.50, 0.45, 0.40, 0.60],
    [0.60, 0,    0.50, 0.40, 0.60, 0.50, 0.55, 0.35, 0.50, 0.45, 0.50, 0.55, 0.40, 0.35, 0.50],
    [0.50, 0.50, 0,    0.60, 0.50, 0.45, 0.35, 0.55, 0.45, 0.50, 0.40, 0.45, 0.60, 0.55, 0.45],
    [0.40, 0.40, 0.60, 0,    0.50, 0.45, 0.30, 0.40, 0.55, 0.45, 0.40, 0.45, 0.50, 0.45, 0.40],
    [0.70, 0.60, 0.50, 0.50, 0,    0.60, 0.50, 0.60, 0.65, 0.60, 0.55, 0.60, 0.50, 0.55, 0.60],
    [0.55, 0.50, 0.45, 0.45, 0.60, 0,    0.65, 0.40, 0.70, 0.60, 0.75, 0.70, 0.55, 0.50, 0.70],
    [0.45, 0.55, 0.35, 0.30, 0.50, 0.65, 0,    0.45, 0.60, 0.65, 0.50, 0.70, 0.40, 0.45, 0.70],
    [0.30, 0.35, 0.55, 0.40, 0.60, 0.40, 0.45, 0,    0.55, 0.50, 0.40, 0.45, 0.60, 0.70, 0.45],
    [0.50, 0.50, 0.45, 0.55, 0.65, 0.70, 0.60, 0.55, 0,    0.70, 0.65, 0.75, 0.55, 0.55, 0.75],
    [0.45, 0.45, 0.50, 0.45, 0.60, 0.60, 0.65, 0.50, 0.70, 0,    0.50, 0.60, 0.55, 0.60, 0.70],
    [0.55, 0.50, 0.40, 0.40, 0.55, 0.75, 0.50, 0.40, 0.65, 0.50, 0,    0.70, 0.60, 0.50, 0.65],
    [0.50, 0.55, 0.45, 0.45, 0.60, 0.70, 0.70, 0.45, 0.75, 0.60, 0.70, 0,    0.55, 0.50, 0.75],
    [0.45, 0.40, 0.60, 0.50, 0.50, 0.55, 0.40, 0.60, 0.55, 0.55, 0.60, 0.55, 0,    0.70, 0.50],
    [0.40, 0.35, 0.55, 0.45, 0.55, 0.50, 0.45, 0.70, 0.55, 0.60, 0.50, 0.50, 0.70, 0,    0.60],
    [0.60, 0.50, 0.45, 0.40, 0.60, 0.70, 0.70, 0.45, 0.75, 0.70, 0.65, 0.75, 0.50, 0.60, 0   ]
], dtype=float)

A = A_raw * K_SCALE

s_star = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
                   0.90, 0.90, 0.95, 0.95, 0.95, 0.95,
                   0.90, 0.90, 0.95])

def sigma(x):
    return 1.0 / (1.0 + np.exp(-4.0 * (x - 0.5)))

def sigma_prime(x):
    s = sigma(x)
    return 4.0 * s * (1.0 - s)

def inv_sigma(y):
    y = np.clip(y, 1e-7, 1.0 - 1e-7)
    return 0.5 + np.log(y / (1.0 - y)) / 4.0

def make_b(Amat, s_eq, alpha=ALPHA):
    """Compute b so that s_eq is an exact fixed point of (Amat, b, alpha)."""
    return inv_sigma(s_eq) - (Amat - alpha * np.eye(N_NODES)).dot(s_eq)

def step_update(s, Amat, b_vec, alpha=ALPHA):
    return sigma(Amat.dot(s) + b_vec - alpha * s)

def find_fixed_point(Amat, b_vec, alpha=ALPHA, init=None, tol=1e-12, max_iter=5000):
    """Iterate to find actual fixed point of (Amat, b_vec, alpha)."""
    s = np.ones(N_NODES) * 0.5 if init is None else init.copy()
    for _ in range(max_iter):
        s_new = step_update(s, Amat, b_vec, alpha)
        if np.linalg.norm(s_new - s) < tol:
            return s_new
        s = s_new
    return s

b_nominal = make_b(A, s_star)

# ── Lyapunov construction ────────────────────────────────────────────────────
def build_jacobian(Amat, s_eq, b_vec, alpha=ALPHA):
    """J = diag(sigma'(pre-activation)) @ (A - alpha*I)"""
    pre = Amat.dot(s_eq) + b_vec - alpha * s_eq
    D = np.diag(sigma_prime(pre))
    return D @ (Amat - alpha * np.eye(N_NODES))

def build_lyapunov_P(J):
    """
    Solve discrete Lyapunov equation: J^T P J - P = -I
    Returns P (positive definite if system is stable), or None if it fails.
    """
    try:
        P = solve_discrete_lyapunov(J.T, np.eye(N_NODES))
        eigvals = np.linalg.eigvalsh(P)
        is_pd = bool(np.all(eigvals > 0))
        return P, eigvals, is_pd
    except Exception as e:
        return None, None, False

def V(s, s_eq, P):
    """Lyapunov function value."""
    d = s - s_eq
    return float(d @ P @ d)

def delta_V(s_prev, s_next, s_eq, P):
    """ΔV = V(s_{t+1}) - V(s_t)"""
    return V(s_next, s_eq, P) - V(s_prev, s_eq, P)

# ── Nominal Lyapunov certificate ─────────────────────────────────────────────
def verify_certificate(Amat, b_vec, s_eq, P, n_samples=2000):
    """
    Sample random states, check ΔV < 0 at each.
    Returns fraction satisfying certificate.
    """
    violations = 0
    for _ in range(n_samples):
        s = np.random.rand(N_NODES)
        s_next = step_update(s, Amat, b_vec)
        dv = delta_V(s, s_next, s_eq, P)
        if dv >= 0 and np.linalg.norm(s - s_eq) > 1e-6:
            violations += 1
    return 1.0 - violations / n_samples

# ─────────────────────────────────────────────────────────────────────────────
# FM1: SLOW DEGRADATION
# Sustained per-axis suppression — multiply one axis by (1-rate) each step.
# Track V and ΔV over time. ΔV slowing toward 0 is the early warning.
# ─────────────────────────────────────────────────────────────────────────────
def fm1_slow_degradation(axes_to_test=None, suppression_rate=0.005, steps=FM_STEPS):
    """
    For each axis: start at s*, apply sustained multiplicative suppression
    at `suppression_rate` per step while system also tries to recover.
    """
    if axes_to_test is None:
        # Test the structurally interesting ones
        axes_to_test = [0, 4, 5, 8, 10, 11, 14]  # Love,Self,Trust,Integrity,Transparency,Accountability,Safety

    J = build_jacobian(A, s_star, b_nominal)
    P, _, _ = build_lyapunov_P(J)

    results = {}
    for axis in axes_to_test:
        s = s_star.copy()
        V_trace    = [V(s, s_star, P)]
        dV_trace   = [0.0]
        state_trace = [s.copy()]
        axis_trace  = [s[axis]]

        for t in range(1, steps + 1):
            # Apply sustained suppression to this axis
            s[axis] *= (1.0 - suppression_rate)
            s[axis] = max(0.0, s[axis])
            # System dynamics
            s_next = step_update(s, A, b_nominal)
            dv = delta_V(s, s_next, s_star, P)
            s = s_next
            V_trace.append(V(s, s_star, P))
            dV_trace.append(dv)
            state_trace.append(s.copy())
            axis_trace.append(s[axis])

        results[NODE_NAMES[axis]] = {
            "V":      np.array(V_trace),
            "dV":     np.array(dV_trace),
            "states": np.array(state_trace),
            "axis_val": np.array(axis_trace),
            "final_V": V_trace[-1],
            "final_axis": axis_trace[-1],
            "dV_min":  float(np.min(dV_trace[1:])),
            "dV_zero_crossing": int(np.argmax(np.array(dV_trace[1:]) >= 0)) if np.any(np.array(dV_trace[1:]) >= 0) else -1,
        }

    return results

# ─────────────────────────────────────────────────────────────────────────────
# FM2: BASELINE EROSION
# b degrades linearly from b_nominal toward b_orig (or toward zero).
# Track V against: (a) fixed nominal s*, (b) actual drifting fixed point.
# ─────────────────────────────────────────────────────────────────────────────
b_orig = np.array([0.05,0.05,0.05,0.05,0.06,0.06,0.04,0.04,0.06,0.05,0.06,0.05,0.05,0.05,0.06])

def fm2_baseline_erosion(steps=FM_STEPS, erode_to="orig"):
    """
    Linearly interpolate b from b_nominal to b_target over `steps` steps.
    At each step, track:
      - V relative to nominal s* (measures drift from ideal)
      - V relative to current actual fixed point (measures local stability)
      - Whether P stays positive definite
    """
    b_target = b_orig if erode_to == "orig" else np.zeros(N_NODES)

    s = s_star.copy()
    J_nom = build_jacobian(A, s_star, b_nominal)
    P_nom, _, _ = build_lyapunov_P(J_nom)

    records = []
    for t in range(steps + 1):
        frac = t / steps
        b_t = b_nominal * (1 - frac) + b_target * frac

        # Current actual fixed point under b_t
        s_fp_t = find_fixed_point(A, b_t, init=s_star)

        # Lyapunov P under current system
        J_t = build_jacobian(A, s_fp_t, b_t)
        P_t, eigvals_t, is_pd_t = build_lyapunov_P(J_t)

        V_vs_nominal  = V(s, s_star, P_nom)
        V_vs_current  = V(s, s_fp_t, P_t) if is_pd_t else np.nan
        P_min_eig     = float(np.min(eigvals_t)) if eigvals_t is not None else np.nan

        records.append({
            "step":           t,
            "b_mean":         float(b_t.mean()),
            "fp_mean":        float(s_fp_t.mean()),
            "s_mean":         float(s.mean()),
            "V_nominal":      V_vs_nominal,
            "V_current":      V_vs_current,
            "P_min_eig":      P_min_eig,
            "P_is_pd":        int(is_pd_t),
            "attractor_gap":  float(np.linalg.norm(s_fp_t - s_star)),
        })

        # Step dynamics under eroded b
        s = step_update(s, A, b_t)

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# FM3: WEIGHT ASYMMETRY INJECTION
# Adversarially modify off-diagonal weights over time.
# Recompute P at each step. Flag when P loses positive definiteness.
# ─────────────────────────────────────────────────────────────────────────────
def fm3_weight_asymmetry(steps=FM_STEPS, injection_rate=0.003, mode="targeted"):
    """
    Two modes:
      "random"   — add small random asymmetric noise each step
      "targeted" — systematically reduce outgoing weights FROM Autonomy (axis 7),
                   the structurally weakest node, to stress the system's weakest joint
    """
    A_t = A.copy()
    s = s_star.copy()

    records = []
    for t in range(steps + 1):
        b_t = make_b(A_t, s_star)   # recompute b to track moving fixed point
        s_fp_t = find_fixed_point(A_t, b_t, init=s_star)
        J_t = build_jacobian(A_t, s_fp_t, b_t)
        P_t, eigvals_t, is_pd_t = build_lyapunov_P(J_t)

        sr_t = float(max(abs(np.linalg.eigvals(J_t))))
        P_min_eig = float(np.min(eigvals_t)) if eigvals_t is not None else np.nan

        records.append({
            "step":        t,
            "jacobian_sr": sr_t,
            "P_min_eig":   P_min_eig,
            "P_is_pd":     int(is_pd_t),
            "A_max_asym":  float(np.max(np.abs(A_t - A_t.T))),
            "fp_mean":     float(s_fp_t.mean()),
            "s_mean":      float(s.mean()),
            "V":           V(s, s_fp_t, P_t) if is_pd_t else np.nan,
        })

        if t < steps:
            # Inject asymmetry
            if mode == "random":
                noise = np.random.normal(0, injection_rate, A_t.shape)
                noise = noise - noise.T   # purely antisymmetric
                A_t = np.clip(A_t + noise, 0, 1)
                np.fill_diagonal(A_t, 0)
            elif mode == "targeted":
                # Reduce Autonomy's outgoing influence (row 7) and
                # increase incoming (col 7) — creates asymmetric dependency
                A_t[7, :] = np.clip(A_t[7, :] - injection_rate, 0, 1)
                A_t[:, 7] = np.clip(A_t[:, 7] + injection_rate * 0.5, 0, 1)
                np.fill_diagonal(A_t, 0)

            s = step_update(s, A_t, b_t)

    return pd.DataFrame(records), A_t

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(fm1_results, df_fm2, df_fm3, df_fm3_rand, P_nom, P_eigvals):
    sns.set(style="whitegrid", context="talk")

    # ── Lyapunov P heatmap ──
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Lyapunov Matrix P (system-derived contraction metric)", fontsize=13, fontweight='bold')
    sns.heatmap(P_nom, ax=axs[0], cmap="RdBu_r", center=0,
                xticklabels=NODE_NAMES, yticklabels=NODE_NAMES,
                cbar_kws={"label": "P value"})
    axs[0].set_title("P matrix")
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(axs[0].get_yticklabels(), rotation=0, fontsize=7)
    axs[1].plot(sorted(P_eigvals), 'o-', color="#2ca02c")
    axs[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axs[1].set_title(f"P eigenvalues (all > 0 = positive definite ✓)\nmin={P_eigvals.min():.4f}")
    axs[1].set_xlabel("Index (sorted)"); axs[1].set_ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lyapunov_P.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── FM1: Slow degradation — V and ΔV traces ──
    axes_shown = list(fm1_results.keys())
    fig, axs = plt.subplots(2, len(axes_shown), figsize=(4*len(axes_shown), 8))
    fig.suptitle("FM1: Slow Degradation — Lyapunov V and ΔV per Axis", fontsize=13, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(axes_shown)))
    for col, (name, res) in enumerate(fm1_results.items()):
        ax_v  = axs[0][col]
        ax_dv = axs[1][col]
        ax_v.plot(res["V"], color=colors[col], linewidth=1.5)
        ax_v.set_title(f"{name}", fontsize=9)
        ax_v.set_ylabel("V(s)" if col == 0 else "")
        ax_v.set_xlabel("Step")
        ax_dv.plot(res["dV"][1:], color=colors[col], linewidth=1.2, alpha=0.8)
        ax_dv.axhline(0, color='red', linestyle='--', linewidth=1)
        ax_dv.set_ylabel("ΔV" if col == 0 else "")
        ax_dv.set_xlabel("Step")
        zc = res["dV_zero_crossing"]
        if zc > 0:
            ax_dv.axvline(zc, color='orange', linestyle=':', linewidth=2,
                          label=f"ΔV≥0 @ {zc}")
            ax_dv.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fm1_degradation.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # FM1 summary — final V by axis
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(fm1_results.keys())
    final_Vs = [fm1_results[n]["final_V"] for n in names]
    zcs = [fm1_results[n]["dV_zero_crossing"] for n in names]
    colors_bar = ["#C44E52" if zc >= 0 else "#2ca02c" for zc in zcs]
    ax.bar(names, final_Vs, color=colors_bar)
    ax.set_title("FM1: Final Lyapunov V after sustained suppression\n(red = ΔV crossed zero — degradation outpaced recovery)", fontsize=11)
    ax.set_ylabel("Final V(s)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fm1_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── FM2: Baseline erosion ──
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FM2: Baseline Erosion — b degrades from nominal to b_orig", fontsize=13, fontweight='bold')
    axs[0,0].plot(df_fm2["step"], df_fm2["b_mean"], color="#4C72B0")
    axs[0,0].set_title("Mean baseline b over time"); axs[0,0].set_ylabel("b mean")
    axs[0,1].plot(df_fm2["step"], df_fm2["fp_mean"], color="#DD8452", label="Actual FP")
    axs[0,1].axhline(s_star.mean(), color='black', linestyle='--', linewidth=1, label='Nominal s*')
    axs[0,1].set_title("Fixed point drift"); axs[0,1].set_ylabel("FP mean"); axs[0,1].legend()
    axs[1,0].plot(df_fm2["step"], df_fm2["V_nominal"], color="#C44E52", label="V vs nominal s*")
    axs[1,0].plot(df_fm2["step"], df_fm2["V_current"].ffill(), color="#2ca02c", label="V vs current FP")
    axs[1,0].set_title("Lyapunov V: nominal vs current FP"); axs[1,0].set_ylabel("V"); axs[1,0].legend()
    axs[1,1].plot(df_fm2["step"], df_fm2["P_min_eig"], color="#9467bd")
    axs[1,1].axhline(0, color='red', linestyle='--', linewidth=1, label='P definiteness boundary')
    axs[1,1].set_title("P minimum eigenvalue (>0 = certificate valid)"); axs[1,1].set_ylabel("min eig(P)"); axs[1,1].legend()
    for ax in axs.flat: ax.set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fm2_erosion.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── FM3: Weight asymmetry ──
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FM3: Weight Asymmetry Injection", fontsize=13, fontweight='bold')
    for df_, label, col in [(df_fm3, "Targeted (Autonomy)", "#C44E52"),
                             (df_fm3_rand, "Random antisymmetric", "#4C72B0")]:
        axs[0,0].plot(df_["step"], df_["jacobian_sr"], color=col, label=label)
        axs[0,1].plot(df_["step"], df_["P_min_eig"],   color=col, label=label)
        axs[1,0].plot(df_["step"], df_["A_max_asym"],  color=col, label=label)
        axs[1,1].plot(df_["step"], df_["fp_mean"],     color=col, label=label)
    axs[0,0].axhline(1.0, color='red', linestyle='--', linewidth=1)
    axs[0,0].set_title("Jacobian spectral radius"); axs[0,0].set_ylabel("sr(J)")
    axs[0,1].axhline(0, color='red', linestyle='--', linewidth=1)
    axs[0,1].set_title("P min eigenvalue"); axs[0,1].set_ylabel("min eig(P)")
    axs[1,0].set_title("Max asymmetry in A"); axs[1,0].set_ylabel("|A - A^T| max")
    axs[1,1].axhline(s_star.mean(), color='black', linestyle='--', linewidth=1, label='Nominal s*')
    axs[1,1].set_title("Fixed point mean"); axs[1,1].set_ylabel("FP mean")
    for ax in axs.flat:
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fm3_asymmetry.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── FM1 state heatmap — where does each axis end up? ──
    fig, ax = plt.subplots(figsize=(14, 5))
    final_states = np.array([fm1_results[n]["states"][-1] for n in axes_shown])
    sns.heatmap(final_states, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=NODE_NAMES, yticklabels=axes_shown,
                ax=ax, vmin=0.5, vmax=1.0)
    ax.set_title("FM1: Final state of all nodes when each axis is suppressed", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fm1_state_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RELATIONAL SYSTEM — LYAPUNOV ANALYSIS")
    print("=" * 60)

    # ── Build nominal Lyapunov certificate ──
    J_nom = build_jacobian(A, s_star, b_nominal)
    P_nom, P_eigvals, P_is_pd = build_lyapunov_P(J_nom)

    sr_J = float(max(abs(np.linalg.eigvals(J_nom))))
    print(f"\nJacobian sr at s*       : {sr_J:.4f}")
    print(f"P positive definite     : {P_is_pd}  (required for certificate)")
    print(f"P min eigenvalue        : {P_eigvals.min():.4f}")
    print(f"P max eigenvalue        : {P_eigvals.max():.4f}")
    print(f"P condition number      : {P_eigvals.max()/P_eigvals.min():.2f}")

    # Verify certificate over random samples
    print(f"\nVerifying ΔV < 0 over 2000 random states…")
    cert_frac = verify_certificate(A, b_nominal, s_star, P_nom)
    print(f"Certificate holds       : {cert_frac*100:.2f}% of samples")

    # ── FM1: Slow degradation ──
    print("\n── FM1: Slow degradation (rate=0.005/step, 500 steps)…")
    fm1_results = fm1_slow_degradation(suppression_rate=0.005)
    print(f"{'Axis':15s}  {'Final V':>10s}  {'ΔV zero-cross':>14s}  {'Final axis val':>14s}")
    for name, res in fm1_results.items():
        zc_str = f"step {res['dV_zero_crossing']}" if res['dV_zero_crossing'] >= 0 else "never"
        print(f"{name:15s}  {res['final_V']:>10.4f}  {zc_str:>14s}  {res['final_axis']:>14.4f}")

    # ── FM2: Baseline erosion ──
    print("\n── FM2: Baseline erosion (b_nominal → b_orig over 500 steps)…")
    df_fm2 = fm2_baseline_erosion(steps=FM_STEPS, erode_to="orig")
    fp_final = df_fm2.iloc[-1]["fp_mean"]
    V_nom_final = df_fm2.iloc[-1]["V_nominal"]
    P_pd_final = bool(df_fm2.iloc[-1]["P_is_pd"])
    print(f"Initial FP mean         : {df_fm2.iloc[0]['fp_mean']:.4f}  (= s* = 0.934)")
    print(f"Final FP mean           : {fp_final:.4f}  (natural attractor)")
    print(f"Final V vs nominal s*   : {V_nom_final:.4f}")
    print(f"P stays positive def    : {bool(df_fm2['P_is_pd'].all())}")
    print(f"Attractor gap at end    : {df_fm2.iloc[-1]['attractor_gap']:.4f}")

    # ── FM3: Weight asymmetry ──
    print("\n── FM3: Weight asymmetry injection (500 steps)…")
    df_fm3_targ, A_final_targ = fm3_weight_asymmetry(steps=FM_STEPS, mode="targeted")
    df_fm3_rand, A_final_rand = fm3_weight_asymmetry(steps=FM_STEPS, mode="random")
    for df_, label in [(df_fm3_targ, "Targeted"), (df_fm3_rand, "Random")]:
        sr_final = df_.iloc[-1]["jacobian_sr"]
        pd_final = bool(df_.iloc[-1]["P_is_pd"])
        asym_final = df_.iloc[-1]["A_max_asym"]
        fp_final_3 = df_.iloc[-1]["fp_mean"]
        print(f"  {label:10s}: sr={sr_final:.4f}  P_pd={pd_final}  "
              f"max_asym={asym_final:.4f}  FP_mean={fp_final_3:.4f}")

    # ── Plots ──
    print("\n── Generating plots…")
    make_plots(fm1_results, df_fm2, df_fm3_targ, df_fm3_rand, P_nom, P_eigvals)

    # ── Save diagnostics ──
    diag = {
        "lyapunov": {
            "jacobian_sr":       sr_J,
            "P_positive_def":    P_is_pd,
            "P_min_eig":         float(P_eigvals.min()),
            "P_max_eig":         float(P_eigvals.max()),
            "P_condition_number":float(P_eigvals.max()/P_eigvals.min()),
            "certificate_frac":  cert_frac,
        },
        "fm1_degradation": {
            name: {
                "final_V":            res["final_V"],
                "dV_zero_crossing":   res["dV_zero_crossing"],
                "final_axis_val":     float(res["final_axis"]),
            } for name, res in fm1_results.items()
        },
        "fm2_erosion": {
            "final_fp_mean":     float(fp_final),
            "final_V_nominal":   float(V_nom_final),
            "P_always_pd":       bool(df_fm2["P_is_pd"].all()),
            "final_attractor_gap": float(df_fm2.iloc[-1]["attractor_gap"]),
        },
        "fm3_asymmetry": {
            mode: {
                "final_sr":      float(df_.iloc[-1]["jacobian_sr"]),
                "final_P_pd":    bool(df_.iloc[-1]["P_is_pd"]),
                "final_max_asym":float(df_.iloc[-1]["A_max_asym"]),
                "final_fp_mean": float(df_.iloc[-1]["fp_mean"]),
            }
            for mode, df_ in [("targeted", df_fm3_targ), ("random", df_fm3_rand)]
        }
    }
    with open(os.path.join(OUTPUT_DIR, "lyapunov_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    df_fm2.to_csv(os.path.join(OUTPUT_DIR, "fm2_erosion.csv"), index=False)
    df_fm3_targ.to_csv(os.path.join(OUTPUT_DIR, "fm3_targeted.csv"), index=False)
    df_fm3_rand.to_csv(os.path.join(OUTPUT_DIR, "fm3_random.csv"), index=False)

    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")
    print("Done.")
