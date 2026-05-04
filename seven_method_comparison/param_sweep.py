"""
param_sweep.py – Compare 7 MDP solvers on example.py environments
===================================================================
使用 example.py 提供的 4 种 MDP（chain / gambler / gridworld / random），
通过 MDPAdapter 统一接口后，对比 7 种方法的收敛速度与最终值函数质量。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time, random

# ── 从 example.py 获取 MDP ────────────────────────────────────────────────────
from example import get_mdp

# ── 全局画图风格 ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.35, "grid.color": "#d0d0d0",
    "xtick.direction": "out", "ytick.direction": "out",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc",
})

# Wong colorblind-safe palette（7色）
COLORS = ["#0072B2","#E69F00","#009E73","#D55E00","#CC79A7","#56B4E9","#000000"]
MARKERS = ["o","s","^","D","v","P","X"]

def panel_label(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

DISCOUNT_FACTOR = 0.9
THETA = 1e-4

# ══════════════════════════════════════════════════════════════════════════════
# MDPAdapter：将 example.py 的 MDP 转为统一接口 env.P[s][a]
# ══════════════════════════════════════════════════════════════════════════════
class MDPAdapter:
    def __init__(self, mdp):
        self.nS    = mdp.n_states
        self.nA    = mdp.n_actions
        self.shape = (1, mdp.n_states)
        self.terminal_states = self._find_terminal_states(mdp)
        self.P     = self._build(mdp)

    def _find_terminal_states(self, mdp):
        terminals = set()
        for s in range(self.nS):
            is_absorbing = True
            zero_cost = True
            for a in range(self.nA):
                row = mdp.P[s, a]
                if not np.isclose(row[s], 1.0) or not np.isclose(row.sum(), 1.0):
                    is_absorbing = False
                if not np.isclose(mdp.C[s, a], 0.0):
                    zero_cost = False
            if is_absorbing and zero_cost:
                terminals.add(s)
        return terminals

    def _build(self, mdp):
        P = []
        for s in range(self.nS):
            row = []
            for a in range(self.nA):
                trans = [
                    (mdp.P[s, a, ns], ns, -mdp.C[s, a], ns in self.terminal_states)
                    for ns in range(self.nS) if mdp.P[s, a, ns] > 0
                ]
                row.append(trans if trans else [(1.0, s, 0.0, False)])
            P.append(row)
        return P

# ══════════════════════════════════════════════════════════════════════════════
# 7 种算法（均返回 errors 列表和 times 列表）
# ══════════════════════════════════════════════════════════════════════════════
def _qval(s, V, env):
    vals = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, ns, r, _ in env.P[s][a]:
            vals[a] += prob * (r + DISCOUNT_FACTOR * V[ns])
    return vals

def _bellman(V, env):
    Vn = np.empty(env.nS)
    for s in range(env.nS):
        Vn[s] = np.max(_qval(s, V, env))
    return Vn

def _sample_transition(transitions):
    probs = [t[0] for t in transitions]
    idx = np.random.choice(len(transitions), p=probs)
    return transitions[idx]

def get_vstar(env, tol=1e-10):
    V = np.zeros(env.nS)
    for _ in range(10000):
        Vn = _bellman(V, env)
        if np.max(np.abs(Vn - V)) < tol: return Vn
        V = Vn
    return V

# ── Approach 1: VI ────────────────────────────────────────────────────────────
def run_vi(env, Vs, n_iter):
    V = np.zeros(env.nS); errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        V = _bellman(V, env)
        errs.append(np.max(np.abs(V - Vs)))
        ts.append(time.perf_counter() - t0)
    return errs, ts

# ── Approach 2: RandomVI ──────────────────────────────────────────────────────
def run_random_vi(env, Vs, n_iter, frac=0.3):
    V = np.zeros(env.nS); errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        Bk = random.sample(range(env.nS), max(1, int(env.nS * frac)))
        for s in Bk:
            V[s] = np.max(_qval(s, V, env))
        errs.append(np.max(np.abs(V - Vs)))
        ts.append(time.perf_counter() - t0)
    return errs, ts

# ── Approach 3: InfluenceTreeVI ───────────────────────────────────────────────
def run_influence_vi(env, Vs, n_iter, seed_frac=0.2):
    V = np.zeros(env.nS); errs, ts = [], []
    t0 = time.perf_counter()
    B = random.sample(range(env.nS), max(1, int(env.nS * seed_frac)))
    for _ in range(n_iter):
        for s in B:
            V[s] = np.max(_qval(s, V, env))
        influenced = set()
        for s in B:
            for a in range(env.nA):
                for prob, ns, r, _ in env.P[s][a]:
                    if prob > 0: influenced.add(ns)
        B = list(influenced) or random.sample(range(env.nS), max(1, int(env.nS * seed_frac)))
        errs.append(np.max(np.abs(V - Vs)))
        ts.append(time.perf_counter() - t0)
    return errs, ts

# ── Approach 4: CyclicVI ──────────────────────────────────────────────────────
def run_cyclic_vi(env, Vs, n_iter):
    V = np.zeros(env.nS); errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = V.copy()
        for s in range(env.nS):
            y[s] = np.max(_qval(s, y, env))
        V = y
        errs.append(np.max(np.abs(V - Vs)))
        ts.append(time.perf_counter() - t0)
    return errs, ts

# ── Approach 5: RPCyclicVI ────────────────────────────────────────────────────
def run_rp_cyclic_vi(env, Vs, n_iter):
    V = np.zeros(env.nS); errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = V.copy()
        for s in np.random.permutation(env.nS):
            y[s] = np.max(_qval(s, y, env))
        V = y
        errs.append(np.max(np.abs(V - Vs)))
        ts.append(time.perf_counter() - t0)
    return errs, ts

# ── Policy Iteration ──────────────────────────────────────────────────────────
def run_policy_iter(env, Vs, n_iter):
    V      = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype=int)
    errs, ts = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        # policy evaluation (fixed-point)
        for _ in range(200):
            Vn = np.array([
                sum(prob * (r + DISCOUNT_FACTOR * V[ns])
                    for prob, ns, r, _ in env.P[s][policy[s]])
                for s in range(env.nS)
            ])
            if np.max(np.abs(Vn - V)) < 1e-6: V = Vn; break
            V = Vn
        # policy improvement
        for s in range(env.nS):
            policy[s] = np.argmax(_qval(s, V, env))
        errs.append(np.max(np.abs(V - Vs)))
        ts.append(time.perf_counter() - t0)
    return errs, ts

# ── Q-Learning ────────────────────────────────────────────────────────────────
def run_q_learning(
    env,
    Vs,
    n_iter,
    eps_start=0.8,
    eps_end=0.01,
    alpha_start=0.35,
    alpha_end=0.05,
    steps_scale=12,
):
    Q = np.zeros((env.nS, env.nA))
    visit_counts = np.zeros((env.nS, env.nA), dtype=int)
    errs, ts = [], []
    t0 = time.perf_counter()

    nonterminal_states = [s for s in range(env.nS) if s not in env.terminal_states]
    if not nonterminal_states:
        nonterminal_states = list(range(env.nS))

    # Normalize Q-learning work per outer iteration to the state-space size.
    steps_per_iter = max(env.nS, int(np.ceil(steps_scale * env.nS)))
    max_episode_len = max(20, 4 * env.nS)
    state = random.choice(nonterminal_states)
    episode_len = 0

    for it in range(n_iter):
        frac = it / max(1, n_iter - 1)
        eps = eps_end + (eps_start - eps_end) * ((1.0 - frac) ** 2)
        alpha_floor = alpha_end + (alpha_start - alpha_end) * ((1.0 - frac) ** 2)
        for _ in range(steps_per_iter):
            if np.random.rand() < eps:
                action = np.random.randint(env.nA)
            else:
                action = int(np.argmax(Q[state]))

            _, next_state, reward, done = _sample_transition(env.P[state][action])
            visit_counts[state, action] += 1
            alpha = max(alpha_floor, visit_counts[state, action] ** -0.5)
            target = reward if done else reward + DISCOUNT_FACTOR * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])

            episode_len += 1
            if done or episode_len >= max_episode_len:
                state = random.choice(nonterminal_states)
                episode_len = 0
            else:
                state = next_state

        V_q = np.max(Q, axis=1)
        errs.append(np.max(np.abs(V_q - Vs)))
        ts.append(time.perf_counter() - t0)
    return errs, ts

# ══════════════════════════════════════════════════════════════════════════════
# 方法注册表
# ══════════════════════════════════════════════════════════════════════════════
METHODS = [
    ("VI",              run_vi,            COLORS[0], MARKERS[0]),
    ("RandomVI",        run_random_vi,     COLORS[1], MARKERS[1]),
    ("InfluenceTreeVI", run_influence_vi,  COLORS[2], MARKERS[2]),
    ("CyclicVI",        run_cyclic_vi,     COLORS[3], MARKERS[3]),
    ("RPCyclicVI",      run_rp_cyclic_vi,  COLORS[4], MARKERS[4]),
    ("PolicyIter",      run_policy_iter,   COLORS[5], MARKERS[5]),
    ("Q-Learning",      run_q_learning,    COLORS[6], MARKERS[6]),
]
METHOD_ITERS = {
    "VI": 300,
    "RandomVI": 300,
    "InfluenceTreeVI": 300,
    "CyclicVI": 300,
    "RPCyclicVI": 300,
    "PolicyIter": 300,
    "Q-Learning": 300,
}

def iters_to_eps(errs, eps):
    for k, e in enumerate(errs):
        if e <= eps: return k + 1
    return len(errs)

# ══════════════════════════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════════════════════════
MDP_CONFIGS = [
    ("chain",     dict(n=30,  p=0.9,  gamma=0.9)),
    ("gambler",   dict(goal=20, p_win=0.4, gamma=0.9)),
    ("gridworld", dict(k=5,   gamma=0.9, slip_prob=0.1)),
    ("random",    dict(n_states=30, n_actions=4, gamma=0.9, density=0.2, seed=7)),
]
SENSITIVITY_CONFIGS = {
    "chain": {
        "param": "p",
        "values": [0.55, 0.65, 0.75, 0.85, 0.95],
        "base_kwargs": dict(n=30, gamma=0.9),
        "xlabel": "Transition probability $p$",
    },
    "gambler": {
        "param": "p_win",
        "values": [0.25, 0.35, 0.45, 0.55, 0.65],
        "base_kwargs": dict(goal=20, gamma=0.9),
        "xlabel": "Winning probability $p_{win}$",
    },
    "gridworld": {
        "param": "slip_prob",
        "values": [0.00, 0.05, 0.10, 0.20, 0.30],
        "base_kwargs": dict(k=5, gamma=0.9),
        "xlabel": "Slip probability",
    },
    "random": {
        "param": "density",
        "values": [0.10, 0.20, 0.40, 0.60, 0.80],
        "base_kwargs": dict(n_states=30, n_actions=4, gamma=0.9, seed=7),
        "xlabel": "Transition density",
    },
}
N_ITER = max(METHOD_ITERS.values())

print("Building envs and computing V* ...")
envs = []
for name, kwargs in MDP_CONFIGS:
    mdp  = get_mdp(name, **kwargs)
    env  = MDPAdapter(mdp)
    Vs   = get_vstar(env)
    envs.append((name, env, Vs))
    print(f"  {name:10s}  nS={env.nS:4d}  nA={env.nA:3d}")

print("Running 7 methods ...")
all_results = {}
for name, env, Vs in envs:
    np.random.seed(0); random.seed(0)
    all_results[name] = {}
    for mname, fn, col, mk in METHODS:
        errs, ts = fn(env, Vs, METHOD_ITERS[mname])
        all_results[name][mname] = {"errs": errs, "times": ts}
    print(f"  {name} done.")

# ══════════════════════════════════════════════════════════════════════════════
# 参数敏感性实验：扫描每个 MDP 的一个关键模型参数
# ══════════════════════════════════════════════════════════════════════════════
print("Running parameter sensitivity study ...")
sensitivity_results = {}
for name, cfg in SENSITIVITY_CONFIGS.items():
    sensitivity_results[name] = {mname: [] for mname, _, _, _ in METHODS}
    for val in cfg["values"]:
        kwargs = dict(cfg["base_kwargs"])
        kwargs[cfg["param"]] = val
        mdp = get_mdp(name, **kwargs)
        env = MDPAdapter(mdp)
        Vs = get_vstar(env)
        np.random.seed(0)
        random.seed(0)
        for mname, fn, _, _ in METHODS:
            errs, _ = fn(env, Vs, METHOD_ITERS[mname])
            sensitivity_results[name][mname].append(errs[-1])
    print(f"  {name} sensitivity done.")

# ══════════════════════════════════════════════════════════════════════════════
# Fig A – 4 x 1  收敛曲线（error vs iteration），每个 MDP 一个子图
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)
letters = ["(a)","(b)","(c)","(d)"]

for ax, (name, env, Vs), let in zip(axes, envs, letters):
    for mname, _, col, mk in METHODS:
        errs = np.maximum(all_results[name][mname]["errs"], 1e-14)
        xvals = range(1, len(errs) + 1)
        ax.semilogy(xvals, errs, color=col,
                    marker=mk, markevery=max(1, len(errs)//6), markersize=4, label=mname)
    ax.axhline(THETA, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|V^k - V^*\|_\infty$") if ax == axes[0] else None
    ax.set_title(f"{name} MDP", fontweight="bold", pad=3)
    ax.set_xlim(1, N_ITER)
    panel_label(ax, let, x=-0.14)

handles = [plt.Line2D([0],[0], color=c, marker=m, markersize=5,
                       linewidth=1.6, label=n)
           for n, _, c, m in METHODS]
fig.legend(handles=handles, loc="lower center", ncol=7,
           bbox_to_anchor=(0.5, -0.16), frameon=True,
           columnspacing=0.8, handlelength=1.4)
fig.savefig("figA_convergence_by_iteration.png")
plt.close()
print("Saved figA_convergence_by_iteration.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig B – 4 x 1  error vs wall-clock time
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)

for ax, (name, env, Vs), let in zip(axes, envs, letters):
    for mname, _, col, mk in METHODS:
        res  = all_results[name][mname]
        errs = np.maximum(res["errs"], 1e-14)
        ax.semilogy(res["times"], errs, color=col,
                    marker=mk, markevery=max(1, len(errs)//6), markersize=4, label=mname)
    ax.axhline(THETA, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel(r"$\|V^k - V^*\|_\infty$") if ax == axes[0] else None
    ax.set_title(f"{name} MDP", fontweight="bold", pad=3)
    panel_label(ax, let, x=-0.14)

fig.legend(handles=handles, loc="lower center", ncol=7,
           bbox_to_anchor=(0.5, -0.16), frameon=True,
           columnspacing=0.8, handlelength=1.4)
fig.savefig("figB_convergence_by_time.png")
plt.close()
print("Saved figB_convergence_by_time.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig C – Iterations-to-threshold heatmap  (methods × MDPs)
# ══════════════════════════════════════════════════════════════════════════════
EPS_LIST  = [1e-1, 1e-2, 1e-3]
mdp_names = [n for n, _, _ in envs]
mnames    = [m[0] for m in METHODS]

fig, axes = plt.subplots(1, len(EPS_LIST), figsize=(11, 3.0), constrained_layout=True)

for ax, eps, let in zip(axes, EPS_LIST, ["(a)","(b)","(c)"]):
    mat = np.array([
        [iters_to_eps(all_results[mdp][m]["errs"], eps) for mdp in mdp_names]
        for m in mnames
    ], dtype=float)
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=1, vmax=N_ITER)
    for i in range(len(mnames)):
        for j in range(len(mdp_names)):
            v   = int(mat[i, j])
            method_cap = METHOD_ITERS[mnames[i]]
            txt = str(v) if v < method_cap else f">{method_cap}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if mat[i,j] > 0.65*N_ITER else "black")
    ax.set_xticks(range(len(mdp_names))); ax.set_xticklabels(mdp_names, fontsize=8)
    ax.set_yticks(range(len(mnames)));   ax.set_yticklabels(mnames, fontsize=8)
    ax.set_title(f"ε = {eps}", fontweight="bold", pad=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Iterations", fontsize=7); cb.ax.tick_params(labelsize=7)
    panel_label(ax, let, x=-0.22)

fig.savefig("figC_iterations_to_tolerance_heatmap.png")
plt.close()
print("Saved figC_iterations_to_tolerance_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig D – Final error bar chart（4 MDP × 7 methods）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
x     = np.arange(len(METHODS))
width = 0.6

for ax, (name, env, Vs), let in zip(axes, envs, letters):
    finals = [all_results[name][m[0]]["errs"][-1] for m in METHODS]
    bars   = ax.bar(x, finals, width, color=[m[2] for m in METHODS],
                    alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in METHODS], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel(r"Final $\|V^k - V^*\|_\infty$") if ax == axes[0] else None
    ax.set_title(f"{name} MDP", fontweight="bold", pad=3)
    ax.grid(axis="y", alpha=0.4)
    panel_label(ax, let, x=-0.14)

fig.savefig("figD_final_error_comparison.png")
plt.close()
print("Saved figD_final_error_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig E – Sensitivity to model parameters
# 用 line plot 展示关键参数变化下的最终误差
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)

for ax, (name, env, Vs), let in zip(axes, envs, letters):
    cfg = SENSITIVITY_CONFIGS[name]
    xvals = cfg["values"]
    for mname, _, col, mk in METHODS:
        yvals = np.maximum(sensitivity_results[name][mname], 1e-14)
        ax.semilogy(xvals, yvals, color=col, marker=mk,
                    markersize=4, linewidth=1.6, label=mname)
    ax.set_yscale("log")
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel(r"Final $\|V^k - V^*\|_\infty$") if ax == axes[0] else None
    ax.set_title(f"{name} MDP", fontweight="bold", pad=3)
    ax.grid(axis="y", alpha=0.4)
    panel_label(ax, let, x=-0.14)

fig.legend(handles=handles, loc="lower center", ncol=7,
           bbox_to_anchor=(0.5, -0.12), frameon=True,
           columnspacing=0.8, handlelength=1.4)
fig.savefig("figE_parameter_sensitivity.png")
plt.close()
print("Saved figE_parameter_sensitivity.png")

print("\nAll sweep figures saved.")
