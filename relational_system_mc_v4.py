"""
relational_system_mc_v4.py

15-node relational system — full research-grade extension.

Adds (from Copilot v3 review):
  EXT 1 — Global stability: basin mapping + KMeans hidden-attractor detection
  EXT 2 — Bifurcation scan: K_SCALE × ALPHA phase diagram
  EXT 3 — Axis dominance: per-axis hard-collapse recovery ranking
  EXT 4 — Catastrophic drift: named multi-axis collapse scenarios
  EXT 5 — Adversarial initial states: corners, zero, one, alternating

Plus all v3 fixes:
  FIX 1-7 from previous pass (see v3 header)

Nodes:
  0 Love, 1 Loyalty, 2 Devotion, 3 Faith, 4 Self,
  5 Trust, 6 Boundaries, 7 Autonomy, 8 Integrity, 9 Resilience,
  10 Transparency, 11 Accountability, 12 Learning, 13 Adaptability, 14 Safety

Dependencies:
  pip install numpy pandas matplotlib seaborn tqdm scikit-learn

Run:
  python relational_system_mc_v4.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from sklearn.cluster import KMeans
import os, json, time

# ── Config ───────────────────────────────────────────────────────────────────
np.random.seed(42)

N_NODES       = 15
ALPHA         = 0.12
EPSILON       = 0.05
K_SUSTAIN     = 5
MAX_STEPS     = 500
N_TRIALS      = 800
PERTURB_PROB  = 0.6
SHOCK_MAG     = 1.0
K_SCALE       = 0.1418
SHOCK_MODE    = "multiplicative"
OUTPUT_DIR    = "mc_results_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODE_NAMES = [
    "Love", "Loyalty", "Devotion", "Faith", "Self",
    "Trust", "Boundaries", "Autonomy", "Integrity", "Resilience",
    "Transparency", "Accountability", "Learning", "Adaptability", "Safety"
]

# ── Matrix ───────────────────────────────────────────────────────────────────
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

b_orig = np.array([0.05, 0.05, 0.05, 0.05, 0.06, 0.06,
                   0.04, 0.04, 0.06, 0.05, 0.06, 0.05,
                   0.05, 0.05, 0.06])

# ── Core dynamics ─────────────────────────────────────────────────────────────
def sigma(x):
    return 1.0 / (1.0 + np.exp(-4.0 * (x - 0.5)))

def sigma_prime(x):
    s = sigma(x)
    return 4.0 * s * (1.0 - s)

def inv_sigma(y):
    y = np.clip(y, 1e-7, 1.0 - 1e-7)
    return 0.5 + np.log(y / (1.0 - y)) / 4.0

b_nominal = inv_sigma(s_star) - (A - ALPHA * np.eye(N_NODES)).dot(s_star)

def step_update(s, Amat, b_vec, alpha=ALPHA):
    return sigma(Amat.dot(s) + b_vec - alpha * s)

def recovery_dist(s, s_target):
    return np.linalg.norm(s - s_target) / np.sqrt(N_NODES)

def jacobian_sr(Amat, s_eq, b_vec, alpha=ALPHA):
    pre = Amat.dot(s_eq) + b_vec - alpha * s_eq
    D = np.diag(sigma_prime(pre))
    J = D @ (Amat - alpha * np.eye(N_NODES))
    return float(max(abs(np.linalg.eigvals(J))))

def apply_shock(s, axis, mag):
    s = s.copy()
    if SHOCK_MODE == "multiplicative":
        s[axis] = max(0.0, s[axis] * (1.0 - mag))
    elif SHOCK_MODE == "additive":
        s[axis] = np.clip(s[axis] - mag * 0.5, 0.0, 1.0)
    else:
        s[axis] = np.random.uniform(0.0, 0.05)
    return s

RECOVERY_EPS = EPSILON

def run_trial(s0, Amat, b_vec, s_target, perturb=None, regime="A", alpha=ALPHA):
    s = s0.copy()
    traj = [s.copy()]
    pm = {}
    for t, ax, mg in (perturb or []):
        pm.setdefault(t, []).append((ax, mg))
    max_dev = recovery_dist(s, s_target)
    rt = None
    sustain = 0
    for t in range(1, MAX_STEPS + 1):
        if t in pm:
            for ax, mg in pm[t]:
                s = apply_shock(s, ax, mg)
        s = step_update(s, Amat, b_vec, alpha)
        traj.append(s.copy())
        dev = recovery_dist(s, s_target)
        if dev > max_dev: max_dev = dev
        if dev <= RECOVERY_EPS:
            sustain += 1
            if sustain >= K_SUSTAIN:
                rt = t - K_SUSTAIN + 1
                break
        else:
            sustain = 0
    return {"traj": np.array(traj), "recovery_time": rt,
            "max_dev": max_dev, "failed": rt is None}

# ─────────────────────────────────────────────────────────────────────────────
# EXT 1: GLOBAL STABILITY — basin mapping + hidden attractor detection
# ─────────────────────────────────────────────────────────────────────────────
def basin_map(num_points=3000, steps=2000):
    """Run num_points random initial states to equilibrium, collect endpoints."""
    endpoints = []
    for _ in trange(num_points, desc="Basin map"):
        s = np.random.rand(N_NODES)
        for _ in range(steps):
            s = step_update(s, A, b_nominal)
        endpoints.append(s)
    return np.array(endpoints)

def detect_attractors(endpoints, max_k=5):
    """
    Try k=1..max_k clusters, pick best by inertia elbow.
    Returns centers for each k and inertias so caller can decide.
    """
    results = {}
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        km.fit(endpoints)
        results[k] = {"centers": km.cluster_centers_, "labels": km.labels_,
                      "inertia": km.inertia_}
    return results

# ─────────────────────────────────────────────────────────────────────────────
# EXT 2: BIFURCATION SCAN — K_SCALE × ALPHA phase diagram
# ─────────────────────────────────────────────────────────────────────────────
def bifurcation_scan(k_values, alpha_values):
    rows = []
    for k in k_values:
        Ak = A_raw * k
        bk = inv_sigma(s_star) - (Ak - alpha_values[0] * np.eye(N_NODES)).dot(s_star)
        for a in alpha_values:
            bk_a = inv_sigma(s_star) - (Ak - a * np.eye(N_NODES)).dot(s_star)
            sr = jacobian_sr(Ak, s_star, bk_a, alpha=a)
            rows.append({"k": round(k, 4), "alpha": round(a, 4), "jacobian_sr": sr,
                         "stable": int(sr < 1.0)})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# EXT 3: AXIS DOMINANCE — per-axis hard-collapse recovery ranking
# ─────────────────────────────────────────────────────────────────────────────
def axis_recovery_test():
    rows = []
    for i, name in enumerate(NODE_NAMES):
        s0 = s_star.copy()
        s0[i] = 0.0   # hard collapse of single axis
        res = run_trial(s0, A, b_nominal, s_star)
        rows.append({
            "axis": name,
            "axis_idx": i,
            "recovery_time": res["recovery_time"] if res["recovery_time"] else MAX_STEPS,
            "max_dev": res["max_dev"],
            "failed": int(res["failed"])
        })
    return pd.DataFrame(rows).sort_values("recovery_time", ascending=False)

# ─────────────────────────────────────────────────────────────────────────────
# EXT 4: CATASTROPHIC DRIFT — named multi-axis collapse scenarios
# ─────────────────────────────────────────────────────────────────────────────
def catastrophic_tests():
    scenarios = [
        ("Self + Integrity",                   [4, 8]),
        ("Self + Integrity + Safety",           [4, 8, 14]),
        ("Trust + Boundaries + Accountability", [5, 6, 11]),
        ("Love + Faith + Autonomy",             [0, 3, 7]),
        ("All load-bearing (Integrity+Trust+Safety+Accountability)", [8, 5, 14, 11]),
        ("Full collapse — all axes",            list(range(N_NODES))),
        ("Love alone zeroed",                   [0]),
        ("Self alone zeroed",                   [4]),
    ]
    rows = []
    for name, axes in scenarios:
        s0 = s_star.copy()
        for ax in axes:
            s0[ax] = 0.0
        res = run_trial(s0, A, b_nominal, s_star)
        rows.append({
            "scenario": name,
            "n_axes_collapsed": len(axes),
            "recovery_time": res["recovery_time"] if res["recovery_time"] else MAX_STEPS,
            "max_dev": round(res["max_dev"], 4),
            "failed": int(res["failed"])
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# EXT 5: ADVERSARIAL INITIAL STATES
# ─────────────────────────────────────────────────────────────────────────────
def adversarial_initial_states():
    rng = np.random.default_rng(99)
    tests = [
        ("All-zero",          np.zeros(N_NODES)),
        ("All-one",           np.ones(N_NODES)),
        ("Alternating 0-1",   np.array([i % 2 for i in range(N_NODES)], dtype=float)),
        ("Alternating 1-0",   np.array([(i+1) % 2 for i in range(N_NODES)], dtype=float)),
        ("Random corner 1",   rng.choice([0.0, 1.0], size=N_NODES).astype(float)),
        ("Random corner 2",   rng.choice([0.0, 1.0], size=N_NODES).astype(float)),
        ("Random corner 3",   rng.choice([0.0, 1.0], size=N_NODES).astype(float)),
        ("All-0.5 (center)",  np.ones(N_NODES) * 0.5),
        ("Near-zero (0.01)",  np.ones(N_NODES) * 0.01),
        ("Near-one (0.99)",   np.ones(N_NODES) * 0.99),
    ]
    rows = []
    for name, s0 in tests:
        res = run_trial(s0, A, b_nominal, s_star)
        rows.append({
            "state": name,
            "initial_mean": round(float(s0.mean()), 3),
            "recovery_time": res["recovery_time"] if res["recovery_time"] else MAX_STEPS,
            "max_dev": round(res["max_dev"], 4),
            "failed": int(res["failed"]),
            "final_mean": round(float(res["traj"][-1].mean()), 4)
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo (v3 structure, both regimes)
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo_swarm():
    results = []
    for i in trange(N_TRIALS, desc="MC trials"):
        if np.random.rand() < 0.5:
            s0 = s_star * (0.5 + 0.5 * np.random.rand(N_NODES))
            s0[0] = np.random.uniform(0.05, 0.35)
        else:
            s0 = np.random.uniform(0.1, 0.8, size=N_NODES)
        perturb = []
        if np.random.rand() < PERTURB_PROB:
            n_sh = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            for _ in range(n_sh):
                perturb.append((np.random.randint(1, 40), np.random.randint(0, N_NODES),
                                SHOCK_MAG * np.random.uniform(0.6, 1.0)))
        for regime, Amat, b_vec in [
            ("A", A, b_nominal),
            ("B", np.clip(A + np.random.normal(0, 0.02, A.shape), -1, 1), b_nominal),
        ]:
            res = run_trial(s0.copy(), Amat, b_vec, s_star, perturb=perturb, regime=regime)
            results.append({
                "trial": i, "regime": regime,
                "recovery_time": res["recovery_time"] if res["recovery_time"] else MAX_STEPS,
                "failed": int(res["failed"]), "max_dev": res["max_dev"],
                "n_shocks": len(perturb),
                "perturb_axes": ",".join(str(p[1]) for p in perturb) if perturb else "",
            })
    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(df_mc, df_axis, df_cat, df_adv, df_bif, km_results, endpoints):
    sns.set(style="whitegrid", context="talk")
    c20 = plt.cm.tab20(np.linspace(0, 1, N_NODES))

    # ── Bifurcation phase diagram ──
    pivot = df_bif.pivot(index="alpha", columns="k", values="jacobian_sr")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, cmap="RdYlGn_r", center=1.0, ax=ax,
                cbar_kws={"label": "Jacobian Spectral Radius"},
                linewidths=0, vmin=0, vmax=2.0)
    ax.set_title("Bifurcation Phase Diagram\n(green=stable sr<1, red=unstable sr>1)", fontsize=12)
    ax.set_xlabel("Scale factor k"); ax.set_ylabel("Leak α")
    # Mark current operating point
    k_ticks = list(pivot.columns)
    a_ticks = list(pivot.index)
    k_idx = np.argmin(abs(np.array(k_ticks) - K_SCALE))
    a_idx = np.argmin(abs(np.array(a_ticks) - ALPHA))
    ax.add_patch(plt.Rectangle((k_idx, a_idx), 1, 1, fill=False, edgecolor='blue', lw=3))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bifurcation_phase.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Attractor detection ──
    inertias = [km_results[k]["inertia"] for k in sorted(km_results)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(range(1, len(inertias)+1), inertias, 'o-', color="#4C72B0")
    axs[0].set_title("KMeans Inertia — Elbow Plot\n(distinct attractors show elbow)")
    axs[0].set_xlabel("k (clusters)"); axs[0].set_ylabel("Inertia")
    # Show endpoint spread via first 2 PCA-like axes (just node 0 vs node 4 for simplicity)
    axs[1].scatter(endpoints[:, 0], endpoints[:, 4], alpha=0.15, s=8, c=km_results[2]["labels"], cmap="Set1")
    axs[1].scatter(km_results[2]["centers"][:, 0], km_results[2]["centers"][:, 4],
                   s=200, c=['red','blue'], marker='*', zorder=10)
    axs[1].set_title("Basin Endpoints: Love vs Self\n(k=2 cluster centers = stars)")
    axs[1].set_xlabel("Love"); axs[1].set_ylabel("Self")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "attractor_detection.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Axis dominance bar chart ──
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_axis = ["#C44E52" if r > 5 else "#4C72B0" for r in df_axis["recovery_time"]]
    ax.barh(df_axis["axis"], df_axis["recovery_time"], color=colors_axis)
    ax.set_title("Axis Dominance: Recovery Time from Single-Axis Hard Collapse\n(longer = more critical)", fontsize=11)
    ax.set_xlabel("Steps to Recovery")
    ax.axvline(df_axis["recovery_time"].median(), color='black', linestyle='--', linewidth=1.5, label='Median')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "axis_dominance.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Catastrophic scenarios ──
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_cat = ["#C44E52" if f else "#4C72B0" for f in df_cat["failed"]]
    bars = ax.barh(df_cat["scenario"], df_cat["recovery_time"], color=colors_cat)
    ax.set_title("Catastrophic Drift Scenarios\n(red = failed to recover)", fontsize=11)
    ax.set_xlabel("Steps to Recovery (cap=500)")
    for bar, md in zip(bars, df_cat["max_dev"]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"dev={md:.2f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "catastrophic_scenarios.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Adversarial initial states ──
    fig, ax = plt.subplots(figsize=(11, 5))
    colors_adv = ["#C44E52" if f else "#2ca02c" for f in df_adv["failed"]]
    ax.barh(df_adv["state"], df_adv["recovery_time"], color=colors_adv)
    ax.set_title("Adversarial Initial States — Recovery Time\n(green=recovered, red=failed)", fontsize=11)
    ax.set_xlabel("Steps to Recovery")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "adversarial_states.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── MC recovery by regime ──
    dfA = df_mc[df_mc.regime == "A"]
    dfB = df_mc[df_mc.regime == "B"]
    fig, axs2 = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Recovery Time: Nominal (A) vs Perturbed Ensemble (B)", fontsize=13, fontweight='bold')
    for ax, dfs, label, col in zip(axs2, [dfA, dfB],
                                    ["Regime A — Nominal", "Regime B — Jittered"],
                                    ["#4C72B0", "#C44E52"]):
        sns.histplot(dfs["recovery_time"], bins=40, kde=True, ax=ax, color=col)
        ax.set_title(label); ax.set_xlabel("Steps to Recovery")
        med = dfs["recovery_time"].median()
        ax.axvline(med, color='black', linestyle='--', lw=1.5, label=f'Median={med:.0f}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "recovery_by_regime.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Sample trajectories ──
    fig, axs3 = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Sample Recovery Trajectories (red = shock)", fontsize=13, fontweight='bold')
    demo = [
        ("Low Love, no shock",
         np.array([0.1,0.6,0.6,0.6,0.7,0.7,0.6,0.5,0.7,0.6,0.7,0.6,0.6,0.6,0.7]), []),
        ("Mid state — Trust zeroed",   np.ones(N_NODES)*0.5, [(10,5,1.0)]),
        ("Near attractor — core shock", s_star*0.7, [(15,8,0.9),(15,4,0.9)]),
    ]
    for col, (title, s0d, pd_) in enumerate(demo):
        ax = axs3[col]
        res_d = run_trial(s0d, A, b_nominal, s_star, perturb=pd_)
        traj = res_d['traj'][:80]
        for ni in range(N_NODES):
            ax.plot(traj[:, ni], color=c20[ni], alpha=0.65, lw=1.3,
                    label=NODE_NAMES[ni] if col==0 else "")
        ax.axhline(0.95, color='black', linestyle='--', lw=1, alpha=0.35)
        for t,_,_ in pd_: ax.axvline(t, color='red', linestyle=':', lw=2, alpha=0.8)
        rt_s = f"Recovered @ {res_d['recovery_time']}" if res_d['recovery_time'] else "..."
        ax.set_title(f"{title}\n{rt_s}", fontsize=10)
        ax.set_xlabel("Step"); ax.set_ylim(-0.02, 1.05)
        if col == 0:
            ax.set_ylabel("State Value")
            ax.legend(fontsize=6, ncol=2, loc='lower right', framealpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_trajectories.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Natural vs target attractor ──
    s_nat = np.ones(N_NODES)*0.5
    for _ in range(3000):
        s_new = step_update(s_nat, A, b_orig)
        if np.linalg.norm(s_new - s_nat) < 1e-12: break
        s_nat = s_new
    fig, ax = plt.subplots(figsize=(12,4))
    x = np.arange(N_NODES); w = 0.35
    ax.bar(x-w/2, s_nat,  w, label='Natural attractor (b_orig)', color='#4C72B0', alpha=0.85)
    ax.bar(x+w/2, s_star, w, label='Target attractor s*',        color='#DD8452', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(NODE_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("State Value")
    ax.set_title("Natural vs Target Attractor", fontsize=11)
    ax.set_ylim(0,1.1); ax.legend(); ax.axhline(0.95, color='black', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "attractor_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RELATIONAL SYSTEM — v4 RESEARCH-GRADE ANALYSIS")
    print("=" * 60)
    print(f"Jacobian sr at s*: {jacobian_sr(A, s_star, b_nominal):.4f}")
    fp_check = step_update(s_star, A, b_nominal)
    print(f"Fixed-point residual: {np.linalg.norm(fp_check - s_star):.2e}\n")

    # EXT 1: Basin mapping
    print("── EXT 1: Global basin mapping (3000 points × 2000 steps)…")
    t0 = time.time()
    endpoints = basin_map(num_points=3000, steps=2000)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Endpoint mean range: [{endpoints.mean(axis=1).min():.4f}, {endpoints.mean(axis=1).max():.4f}]")
    print(f"  Endpoint std across points: {endpoints.std(axis=0).mean():.4f}")
    km_results = detect_attractors(endpoints, max_k=5)
    print("  Inertia by k:", {k: round(km_results[k]['inertia'], 2) for k in km_results})
    print("  k=1 center mean:", km_results[1]['centers'][0].mean().round(4))
    print("  k=2 centers means:", [c.mean().round(4) for c in km_results[2]['centers']])

    # EXT 2: Bifurcation scan
    print("\n── EXT 2: Bifurcation scan…")
    k_vals = np.linspace(0.05, 0.35, 25)
    a_vals = np.linspace(0.05, 0.30, 25)
    df_bif = bifurcation_scan(k_vals, a_vals)
    df_bif.to_csv(os.path.join(OUTPUT_DIR, "bifurcation_scan.csv"), index=False)
    stable_frac = df_bif["stable"].mean()
    print(f"  {stable_frac*100:.1f}% of (k,α) grid is stable")
    boundary = df_bif[df_bif["jacobian_sr"].between(0.9, 1.1)]
    print(f"  Bifurcation boundary region: {len(boundary)} cells with sr ∈ [0.9, 1.1]")

    # EXT 3: Axis dominance
    print("\n── EXT 3: Axis recovery test (single-axis hard collapse)…")
    df_axis = axis_recovery_test()
    df_axis.to_csv(os.path.join(OUTPUT_DIR, "axis_recovery_rank.csv"), index=False)
    print(df_axis.to_string(index=False))

    # EXT 4: Catastrophic drift
    print("\n── EXT 4: Catastrophic scenarios…")
    df_cat = catastrophic_tests()
    df_cat.to_csv(os.path.join(OUTPUT_DIR, "catastrophic_tests.csv"), index=False)
    print(df_cat.to_string(index=False))

    # EXT 5: Adversarial initial states
    print("\n── EXT 5: Adversarial initial states…")
    df_adv = adversarial_initial_states()
    df_adv.to_csv(os.path.join(OUTPUT_DIR, "adversarial_states.csv"), index=False)
    print(df_adv.to_string(index=False))

    # Monte Carlo
    print("\n── Monte Carlo swarm…")
    t0 = time.time()
    df_mc = monte_carlo_swarm()
    print(f"  Done in {time.time()-t0:.1f}s")
    df_mc.to_csv(os.path.join(OUTPUT_DIR, "mc_summary.csv"), index=False)
    for regime in ["A", "B"]:
        dfs = df_mc[df_mc.regime == regime]
        print(f"  Regime {regime}: fail={dfs['failed'].mean():.3f}  rt_med={dfs['recovery_time'].median():.0f}")

    # Plots
    print("\n── Generating plots…")
    make_plots(df_mc, df_axis, df_cat, df_adv, df_bif, km_results, endpoints)

    # Save master diagnostics
    diag = {
        "version": "v4",
        "jacobian_sr": float(jacobian_sr(A, s_star, b_nominal)),
        "fixed_point_residual": float(np.linalg.norm(fp_check - s_star)),
        "basin_map": {
            "n_points": 3000,
            "endpoint_mean_mean": float(endpoints.mean()),
            "endpoint_std_mean": float(endpoints.std(axis=0).mean()),
            "kmeans_inertias": {str(k): round(km_results[k]['inertia'], 4) for k in km_results},
            "k1_center_mean": float(km_results[1]['centers'][0].mean()),
            "k2_center_means": [float(c.mean()) for c in km_results[2]['centers']],
        },
        "bifurcation": {
            "stable_fraction": float(stable_frac),
            "boundary_cells": int(len(boundary)),
        },
        "axis_recovery": df_axis.to_dict(orient="records"),
        "catastrophic": df_cat.to_dict(orient="records"),
        "adversarial": df_adv.to_dict(orient="records"),
        "monte_carlo": {
            regime: {
                "failure_rate": float(df_mc[df_mc.regime==regime]["failed"].mean()),
                "rt_mean": float(df_mc[df_mc.regime==regime]["recovery_time"].mean()),
                "rt_median": float(df_mc[df_mc.regime==regime]["recovery_time"].median()),
            } for regime in ["A","B"]
        }
    }
    with open(os.path.join(OUTPUT_DIR, "diagnostics_v4.json"), "w") as f:
        json.dump(diag, f, indent=2)

    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")
    print("Done.")
