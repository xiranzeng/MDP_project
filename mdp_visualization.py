"""
MDP Visualization – Journal-Quality Figures
DDA4300 Project III: Markov Decision Processes
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
import warnings, time, random
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  Global style  (Nature / Science / OR-style)
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    # font
    "font.family":        "DejaVu Sans",
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "legend.fontsize":    7.5,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    # lines & markers
    "lines.linewidth":    1.6,
    "lines.markersize":   4.5,
    "patch.linewidth":    0.6,
    # axes
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.color":         "#d0d0d0",
    "grid.alpha":         0.7,
    # ticks
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    # figure
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    # legend
    "legend.frameon":       True,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#cccccc",
    "legend.borderpad":     0.4,
    "legend.handlelength":  1.6,
})

# Wong (2011) colorblind-safe 8-color palette
PALETTE = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "pink":    "#CC79A7",
}

METHODS = [
    ("VI",               "#0072B2"),   # blue
    ("RandomVI",         "#E69F00"),   # orange
    ("InfluenceTreeVI",  "#009E73"),   # green
    ("CyclicVI",         "#D55E00"),   # vermilion
    ("RPCyclicVI",       "#CC79A7"),   # pink
]
METHOD_NAMES = [m[0] for m in METHODS]
COLORS       = {m[0]: m[1] for m in METHODS}

# ─── panel label helper ───────────────────────────────────────────────────────
def panel_label(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

# ═══════════════════════════════════════════════════════════════════════════════
#  MDP primitives
# ═══════════════════════════════════════════════════════════════════════════════

def make_mdp(m, n_actions=3, sparsity=0.9, gamma=0.9, seed=None):
    rng = np.random.default_rng(seed)
    P, C = [], []
    for i in range(m):
        Pi, Ci = [], []
        for _ in range(n_actions):
            row = rng.random(m)
            if sparsity > 0:
                row[rng.random(m) < sparsity] = 0.0
            row[i] += 1e-6
            row /= row.sum()
            Pi.append(row)
            Ci.append(rng.random())
        P.append(Pi); C.append(Ci)
    return P, C, gamma


def bellman(y, P, C, gamma):
    m = len(y)
    yn = np.empty(m)
    for i in range(m):
        yn[i] = min(C[i][a] + gamma * P[i][a] @ y for a in range(len(C[i])))
    return yn


def true_opt(P, C, gamma, tol=1e-11, max_iter=8000):
    y = np.zeros(len(P))
    for _ in range(max_iter):
        yn = bellman(y, P, C, gamma)
        if np.max(np.abs(yn - y)) < tol:
            return yn
        y = yn
    return y


# ─── 5 algorithms (return errors list + times list) ──────────────────────────

def run_vi(P, C, gamma, y_star, n_iter):
    y = np.zeros(len(P)); errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = bellman(y, P, C, gamma)
        errs.append(np.max(np.abs(y - y_star)))
        ts.append(time.perf_counter() - t0)
    return errs, ts


def run_random_vi(P, C, gamma, y_star, n_iter, frac=0.3):
    m = len(P); y = np.zeros(m); errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        Bk = np.random.choice(m, max(1, int(m*frac)), replace=False)
        for i in Bk:
            y[i] = min(C[i][a] + gamma * P[i][a] @ y for a in range(len(C[i])))
        errs.append(np.max(np.abs(y - y_star)))
        ts.append(time.perf_counter() - t0)
    return errs, ts


def run_influence_vi(P, C, gamma, y_star, n_iter, seed_frac=0.2):
    m = len(P); y = np.zeros(m); errs, ts = [], []
    t0 = time.perf_counter()
    B = list(np.random.choice(m, max(1, int(m*seed_frac)), replace=False))
    for _ in range(n_iter):
        for i in B:
            y[i] = min(C[i][a] + gamma * P[i][a] @ y for a in range(len(C[i])))
        influenced = set()
        for i in B:
            for a in range(len(P[i])):
                influenced.update(np.where(P[i][a] > 0)[0])
        B = list(influenced) or list(np.random.choice(m, max(1, int(m*seed_frac)), replace=False))
        errs.append(np.max(np.abs(y - y_star)))
        ts.append(time.perf_counter() - t0)
    return errs, ts


def run_cyclic_vi(P, C, gamma, y_star, n_iter):
    m = len(P); y = np.zeros(m); errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for i in range(m):
            y[i] = min(C[i][a] + gamma * P[i][a] @ y for a in range(len(C[i])))
        errs.append(np.max(np.abs(y - y_star)))
        ts.append(time.perf_counter() - t0)
    return errs, ts


def run_rp_cyclic_vi(P, C, gamma, y_star, n_iter):
    m = len(P); y = np.zeros(m); errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for i in np.random.permutation(m):
            y[i] = min(C[i][a] + gamma * P[i][a] @ y for a in range(len(C[i])))
        errs.append(np.max(np.abs(y - y_star)))
        ts.append(time.perf_counter() - t0)
    return errs, ts


RUNNERS = {
    "VI":               run_vi,
    "RandomVI":         run_random_vi,
    "InfluenceTreeVI":  run_influence_vi,
    "CyclicVI":         run_cyclic_vi,
    "RPCyclicVI":       run_rp_cyclic_vi,
}

def iters_to_eps(errs, eps):
    for k, e in enumerate(errs):
        if e <= eps: return k + 1
    return len(errs)

# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-run experiments  (mean ± std over n_runs)
# ═══════════════════════════════════════════════════════════════════════════════
M        = 80
N_ITER   = 150
GAMMA    = 0.9
N_RUNS   = 8          # enough for stable std bands

print(f"Running {N_RUNS} trials each on sparse & dense MDPs (m={M}) …")

def multi_run(sparsity, n_runs, m=M, n_iter=N_ITER, gamma=GAMMA):
    all_errs = {name: [] for name in METHOD_NAMES}
    all_times = {name: [] for name in METHOD_NAMES}
    for run in range(n_runs):
        P, C, g = make_mdp(m, sparsity=sparsity, gamma=gamma, seed=run*17+3)
        y_star = true_opt(P, C, g)
        np.random.seed(run)
        random.seed(run)
        for name in METHOD_NAMES:
            errs, ts = RUNNERS[name](P, C, g, y_star, n_iter)
            all_errs[name].append(errs)
            all_times[name].append(ts)
    # stack → (n_runs, n_iter)
    mean_errs  = {n: np.mean(all_errs[n],  axis=0) for n in METHOD_NAMES}
    std_errs   = {n: np.std( all_errs[n],  axis=0) for n in METHOD_NAMES}
    mean_times = {n: np.mean(all_times[n], axis=0) for n in METHOD_NAMES}
    return mean_errs, std_errs, mean_times

sp_me, sp_se, sp_mt = multi_run(sparsity=0.9,  n_runs=N_RUNS)
de_me, de_se, de_mt = multi_run(sparsity=0.02, n_runs=N_RUNS)
print("  done.")

iters = np.arange(1, N_ITER+1)

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 – Convergence curves: sparse vs dense  (2-panel, journal width)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

for ax, me, se, title, letter in zip(
        axes,
        [sp_me, de_me], [sp_se, de_se],
        ["Sparse transition matrix (density 10%)",
         "Dense transition matrix (density 98%)"],
        ["a", "b"]):
    for name in METHOD_NAMES:
        m_e = np.maximum(me[name], 1e-14)
        s_e = se[name]
        c = COLORS[name]
        ax.semilogy(iters, m_e, color=c, label=name, zorder=3)
        # ±1 std shading (clip to positive for log scale)
        lo = np.maximum(m_e - s_e, 1e-14)
        hi = m_e + s_e
        ax.fill_between(iters, lo, hi, color=c, alpha=0.12, zorder=2)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|y^k - y^*\|_\infty$")
    ax.set_title(title, pad=4)
    ax.set_xlim(1, N_ITER)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
    panel_label(ax, f"({letter})")

# single shared legend below the two panels
handles = [plt.Line2D([0],[0], color=COLORS[n], linewidth=1.8, label=n)
           for n in METHOD_NAMES]
fig.legend(handles=handles, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.14), frameon=True,
           columnspacing=1.0, handlelength=1.6)
fig.savefig("fig1_convergence.pdf")
fig.savefig("fig1_convergence.png")
plt.close()
print("Saved fig1_convergence")

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 – Error vs wall-clock time  (sparse | dense)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

for ax, me, se, mt, title, letter in zip(
        axes,
        [sp_me, de_me], [sp_se, de_se], [sp_mt, de_mt],
        ["Sparse MDP", "Dense MDP"],
        ["a", "b"]):
    for name in METHOD_NAMES:
        t  = mt[name]
        m_e = np.maximum(me[name], 1e-14)
        c = COLORS[name]
        ax.semilogy(t, m_e, color=c, label=name, zorder=3)
        lo = np.maximum(m_e - se[name], 1e-14)
        ax.fill_between(t, lo, m_e + se[name], color=c, alpha=0.12, zorder=2)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel(r"$\|y^k - y^*\|_\infty$")
    ax.set_title(title, pad=4)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
    panel_label(ax, f"({letter})")

fig.legend(handles=handles, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.14), frameon=True,
           columnspacing=1.0, handlelength=1.6)
fig.savefig("fig2_time.pdf")
fig.savefig("fig2_time.png")
plt.close()
print("Saved fig2_time")

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3 – Iterations-to-threshold heatmap  (methods × ε levels)
# ═══════════════════════════════════════════════════════════════════════════════
EPS_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

def iters_matrix(me):
    mat = np.zeros((len(METHOD_NAMES), len(EPS_LIST)))
    for i, name in enumerate(METHOD_NAMES):
        for j, eps in enumerate(EPS_LIST):
            mat[i, j] = iters_to_eps(me[name], eps)
    return mat

mat_sp = iters_matrix(sp_me)
mat_de = iters_matrix(de_me)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
eps_labels = [r"$10^{-1}$", r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$", r"$10^{-5}$"]

for ax, mat, title, letter in zip(
        axes, [mat_sp, mat_de],
        ["Sparse MDP", "Dense MDP"], ["a", "b"]):
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                   vmin=1, vmax=N_ITER)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat[i, j])
            txt = str(val) if val < N_ITER else f">{N_ITER}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color="black" if mat[i,j] < 0.7*N_ITER else "white")
    ax.set_xticks(range(len(EPS_LIST)))
    ax.set_xticklabels(eps_labels, fontsize=8)
    ax.set_yticks(range(len(METHOD_NAMES)))
    ax.set_yticklabels(METHOD_NAMES, fontsize=8)
    ax.set_xlabel("Tolerance $\\varepsilon$")
    ax.set_title(title, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Iterations", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    panel_label(ax, f"({letter})", x=-0.22)

fig.savefig("fig3_heatmap.pdf")
fig.savefig("fig3_heatmap.png")
plt.close()
print("Saved fig3_heatmap")

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 4 – Scalability  (iterations-to-ε=0.01 vs m,  sparse & dense)
# ═══════════════════════════════════════════════════════════════════════════════
M_VALS   = [20, 40, 60, 80, 100, 120]
EPS_SC   = 0.01
N_SC     = 200
N_RUNS_SC = 6

print(f"Scalability experiment (m = {M_VALS}) …")
scale = {"sparse": {n: [] for n in METHOD_NAMES},
         "dense":  {n: [] for n in METHOD_NAMES}}

for m_val in M_VALS:
    print(f"  m = {m_val}")
    for tag, sp in [("sparse", 0.9), ("dense", 0.02)]:
        # average over N_RUNS_SC seeds
        counts = {n: [] for n in METHOD_NAMES}
        for run in range(N_RUNS_SC):
            P, C, g = make_mdp(m_val, sparsity=sp, gamma=GAMMA, seed=run*31+7)
            y_star = true_opt(P, C, g)
            np.random.seed(run); random.seed(run)
            for name in METHOD_NAMES:
                errs, _ = RUNNERS[name](P, C, g, y_star, N_SC)
                counts[name].append(iters_to_eps(errs, EPS_SC))
        for name in METHOD_NAMES:
            scale[tag][name].append(np.mean(counts[name]))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
markers = ["o", "s", "^", "D", "v"]

for ax, tag, title, letter in zip(
        axes, ["sparse", "dense"],
        ["Sparse MDP", "Dense MDP"], ["a", "b"]):
    for (name, _), mk in zip(METHODS, markers):
        ax.plot(M_VALS, scale[tag][name], color=COLORS[name],
                marker=mk, markersize=5, label=name, zorder=3)
    ax.set_xlabel("Number of states $m$")
    ax.set_ylabel(f"Iterations to $\\varepsilon={EPS_SC}$")
    ax.set_title(title, pad=4)
    ax.set_xticks(M_VALS)
    panel_label(ax, f"({letter})")

fig.legend(handles=handles, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.14), frameon=True,
           columnspacing=1.0, handlelength=1.6)
fig.savefig("fig4_scalability.pdf")
fig.savefig("fig4_scalability.png")
plt.close()
print("Saved fig4_scalability")

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 5 – Discount factor γ sensitivity  (VI vs CyclicVI)
# ═══════════════════════════════════════════════════════════════════════════════
GAMMAS = [0.70, 0.80, 0.90, 0.95, 0.99]
gamma_palette = ["#0072B2","#56B4E9","#009E73","#E69F00","#D55E00"]

print("Gamma sensitivity …")
P_g, C_g, _ = make_mdp(60, sparsity=0.85, gamma=0.9, seed=99)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
for ax, name, letter in zip(axes, ["VI", "CyclicVI"], ["a", "b"]):
    for g, col in zip(GAMMAS, gamma_palette):
        y_star_g = true_opt(P_g, C_g, g)
        np.random.seed(0)
        errs, _ = RUNNERS[name](P_g, C_g, g, y_star_g, N_ITER)
        m_e = np.maximum(errs, 1e-14)
        ax.semilogy(iters, m_e, color=col, label=f"$\\gamma={g}$", zorder=3)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|y^k - y^*\|_\infty$")
    ax.set_title(name, pad=4)
    ax.set_xlim(1, N_ITER)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
    panel_label(ax, f"({letter})")
    ax.legend(fontsize=7.5, handlelength=1.4)

fig.savefig("fig5_gamma.pdf")
fig.savefig("fig5_gamma.png")
plt.close()
print("Saved fig5_gamma")

# ═══════════════════════════════════════════════════════════════════════════════
#  Tic-Tac-Toe MDP
# ═══════════════════════════════════════════════════════════════════════════════
print("Solving Tic-Tac-Toe via Value Iteration …")

def check_win(b):
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,bb,c in lines:
        if b[a] == b[bb] == b[c] != 0: return b[a]
    return 0

def is_term(b): return check_win(b) != 0 or 0 not in b

def bfs_states():
    from collections import deque
    visited = {}
    q = deque([((0,)*9, True)])
    while q:
        b, xt = q.popleft()
        if b in visited: continue
        visited[b] = xt
        if is_term(b): continue
        emp = [i for i,v in enumerate(b) if v == 0]
        mark = 1 if xt else -1
        for i in emp:
            nb = b[:i] + (mark,) + b[i+1:]
            q.append((nb, not xt))
    return visited

states_ttt = bfs_states()

def ttt_value(b):
    w = check_win(b)
    if w == 1:  return -1.0
    if w == -1: return  1.0
    return 0.0

V = {b: ttt_value(b) if is_term(b) else 0.0 for b in states_ttt}
deltas = []
for _ in range(500):
    nV = dict(V); d = 0.0
    for b, xt in states_ttt.items():
        if is_term(b): continue
        emp = [i for i,v in enumerate(b) if v == 0]
        if xt:
            nV[b] = min(V.get(b[:i]+(1,)+b[i+1:], 0.0) for i in emp)
        else:
            ns = [b[:i]+(-1,)+b[i+1:] for i in emp]
            nV[b] = sum(V.get(s, 0.0) for s in ns) / len(ns)
        d = max(d, abs(nV[b] - V[b]))
    V = nV; deltas.append(d)
    if d < 1e-10: break

print(f"  Converged in {len(deltas)} iterations.")

# ─── Figure 6 – Tic-Tac-Toe  (3-panel) ───────────────────────────────────────
fig = plt.figure(figsize=(7.2, 2.8), constrained_layout=True)
gs  = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])
ax1 = fig.add_subplot(gs[0])   # first-move heatmap
ax2 = fig.add_subplot(gs[1])   # state-space pie
ax3 = fig.add_subplot(gs[2])   # VI convergence

# (a) first-move heatmap
empty = (0,)*9
vals = []
for i in range(9):
    nb = empty[:i] + (1,) + empty[i+1:]
    vals.append(V.get(nb, 0.0))
grid = np.array(vals).reshape(3,3)

from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax1.imshow(grid, cmap="RdYlGn_r", norm=norm, aspect="equal")
for r in range(3):
    for c in range(3):
        v = grid[r, c]
        ax1.text(c, r, f"{v:.2f}", ha="center", va="center",
                 fontsize=9, fontweight="bold",
                 color="white" if abs(v) > 0.55 else "#222222")
for x in [0.5, 1.5]:
    ax1.axvline(x, color="#555555", linewidth=1.2)
    ax1.axhline(x, color="#555555", linewidth=1.2)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, ticks=[-1,0,1])
cb.ax.set_yticklabels(["X wins","Draw","O wins"], fontsize=6.5)
cb.ax.tick_params(length=0)
ax1.set_title("First-move value (X agent)", pad=4, fontsize=8.5)
panel_label(ax1, "(a)", x=-0.06)

# mark optimal cell
best = int(np.argmin(grid))
br, bc = divmod(best, 3)
ax1.add_patch(plt.Rectangle((bc-0.49, br-0.49), 0.98, 0.98,
              fill=False, edgecolor="#0072B2", linewidth=2.5, zorder=5))

# (b) state-space pie
n_xw  = sum(1 for b in states_ttt if is_term(b) and check_win(b) ==  1)
n_ow  = sum(1 for b in states_ttt if is_term(b) and check_win(b) == -1)
n_dr  = sum(1 for b in states_ttt if is_term(b) and check_win(b) ==  0)
n_nt  = len(states_ttt) - n_xw - n_ow - n_dr
sizes  = [n_xw, n_ow, n_dr, n_nt]
clrs_p = [PALETTE["green"], PALETTE["vermil"], "#888888", PALETTE["sky"]]
lbls_p = ["X wins", "O wins", "Draw", "Non-terminal"]
wedges, texts, autotexts = ax2.pie(
    sizes, labels=lbls_p, colors=clrs_p, autopct="%1.0f%%",
    startangle=110, textprops={"fontsize": 7},
    wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    pctdistance=0.72)
for at in autotexts: at.set_fontsize(6.5)
ax2.set_title(f"State space ($N={len(states_ttt)}$)", pad=4, fontsize=8.5)
panel_label(ax2, "(b)", x=-0.08)

# (c) TTT VI convergence
dl = np.maximum(deltas, 1e-14)
ax3.semilogy(range(1, len(dl)+1), dl, color=PALETTE["blue"], linewidth=1.8)
ax3.set_xlabel("Iteration $k$")
ax3.set_ylabel(r"$\max_s |V^k(s)-V^*(s)|$")
ax3.set_title("VI convergence on TTT", pad=4, fontsize=8.5)
ax3.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
panel_label(ax3, "(c)", x=-0.26)

fig.savefig("fig6_tictactoe.pdf")
fig.savefig("fig6_tictactoe.png")
plt.close()
print("Saved fig6_tictactoe")

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 7 – Summary radar / spider chart
# ═══════════════════════════════════════════════════════════════════════════════
# Metrics: (lower is better for all → invert for radar)
# 1. iterations to eps=0.01 (sparse)  2. iterations to eps=0.01 (dense)
# 3. wall-clock time at iter=N_ITER (sparse)  4. same dense
# 5. final error (sparse)

categories = ["Conv.\n(sparse)", "Conv.\n(dense)",
              "Time\n(sparse)", "Time\n(dense)", "Final\nerror"]
N_cat = len(categories)

# collect raw scores
raw = {}
for name in METHOD_NAMES:
    raw[name] = [
        iters_to_eps(sp_me[name], 0.01),
        iters_to_eps(de_me[name], 0.01),
        sp_mt[name][-1],
        de_mt[name][-1],
        sp_me[name][-1],
    ]

# normalize each metric to [0,1] across methods, then invert (1=best)
raw_arr = np.array([raw[n] for n in METHOD_NAMES])   # (5, 5)
col_min = raw_arr.min(axis=0)
col_max = raw_arr.max(axis=0)
denom = np.where(col_max - col_min < 1e-15, 1.0, col_max - col_min)
normed = 1.0 - (raw_arr - col_min) / denom            # invert: higher = better

angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(3.6, 3.6),
                        subplot_kw={"polar": True})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25","0.50","0.75","1.00"], fontsize=5.5, color="#999999")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=7.5)
ax.spines["polar"].set_linewidth(0.5)
ax.grid(linewidth=0.4, color="#cccccc")

for i, name in enumerate(METHOD_NAMES):
    vals = normed[i].tolist() + [normed[i][0]]
    ax.plot(angles, vals, color=COLORS[name], linewidth=1.6, label=name, zorder=3)
    ax.fill(angles, vals, color=COLORS[name], alpha=0.08)

ax.set_title("Algorithm performance profile\n(outer = better)", pad=14, fontsize=8.5)
ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18),
          fontsize=7, handlelength=1.4, frameon=True)

fig.savefig("fig7_radar.pdf", bbox_inches="tight")
fig.savefig("fig7_radar.png", bbox_inches="tight")
plt.close()
print("Saved fig7_radar")

print("\nAll figures saved (PNG + PDF, 300 dpi).")
