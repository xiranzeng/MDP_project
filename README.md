# Project III – MDP: Five Value Iteration Algorithms
## DDA4300 Optimization Course

---

## Overview

Implements and compares **five Value Iteration (VI) approaches** for solving Markov Decision Processes, plus a Tic-Tac-Toe application solved via MDP. All figures are generated at **300 dpi** in both PNG and PDF format, following top-journal typographic standards (Nature / OR journals).

---

## Algorithms

| # | Name | Description |
|---|------|-------------|
| 1 | **VI** | Standard Value Iteration – full Bellman sweep over all states |
| 2 | **RandomVI** | Randomly selects 30% of states per iteration |
| 3 | **InfluenceTreeVI** | Updates only states reachable via the influence tree; most effective on sparse matrices |
| 4 | **CyclicVI** | Gauss-Seidel style: sweeps states 1→m, reusing freshly updated values immediately |
| 5 | **RPCyclicVI** | Same as CyclicVI with a random permutation order each iteration |

---

## Figures

| File | Description |
|------|-------------|
| `fig1_convergence.png/pdf` | **Convergence curves** ‖y^k − y*‖∞ vs. iteration (mean ± 1 std over 8 runs), sparse vs. dense |
| `fig2_time.png/pdf` | **Error vs. wall-clock time** for all 5 methods, sparse vs. dense |
| `fig3_heatmap.png/pdf` | **Iterations-to-threshold heatmap** – methods × tolerance ε levels |
| `fig4_scalability.png/pdf` | **Scalability** – iterations-to-ε=0.01 as problem size m grows 20→120 |
| `fig5_gamma.png/pdf` | **Discount factor γ sensitivity** for VI and CyclicVI |
| `fig6_tictactoe.png/pdf` | **Tic-Tac-Toe** – first-move value heatmap + state-space distribution + VI convergence |
| `fig7_radar.png/pdf` | **Radar chart** – multi-metric performance profile of all 5 algorithms |

---

## How to Run

```bash
python3 mdp_visualization.py
```

**Requirements:** `numpy`, `matplotlib`

```bash
pip3 install numpy matplotlib
```

Expected runtime: ~3–5 minutes (8 independent trials per experiment).

---

## Key Findings

### Convergence
- **CyclicVI** and **RPCyclicVI** converge in significantly fewer iterations than standard VI, because each sweep reuses freshly updated values (Gauss-Seidel effect). The contraction is tighter per sweep.
- **RandomVI** is slowest per iteration (updates only a fraction of states), but each iteration is cheaper in wall-clock time on large problems.
- **InfluenceTreeVI** achieves the best wall-clock efficiency on **sparse** transition matrices by skipping zero-probability states entirely.

### Effect of Matrix Density
- On **sparse** MDPs (10% nonzero), InfluenceTreeVI and RandomVI gain a clear speed advantage.
- On **dense** MDPs (98% nonzero), CyclicVI/RPCyclicVI dominate on both iteration count and time.

### Discount Factor γ
- Higher γ (→1) slows convergence for all methods; the theoretical bound ‖y^{k+1}−y*‖∞ ≤ γ‖y^k−y*‖∞ is tight.
- CyclicVI retains its advantage across all γ values tested (0.70–0.99).

### Tic-Tac-Toe
- The **center cell** achieves the lowest value (best for X), confirming the classical result.
- Value Iteration on the full state space (~5478 states) converges in **9 iterations**.
- With optimal X vs. uniform-random O, X **never loses**.

---

## Figure Design Notes
- **Colorblind-safe palette**: Wong (2011) 8-color scheme throughout.
- **Uncertainty bands**: ±1 std deviation over 8 independent random seeds.
- **Output**: 300 dpi PNG + vector PDF for publication submission.
- **Layout**: Single-column (3.6 in) or double-column (7.2 in) widths matching Nature/OR journal specs.

---

## File Structure

```
DDA4300/
├── mdp_visualization.py       ← main script
├── README.md                  ← this file
├── fig1_convergence.{png,pdf}
├── fig2_time.{png,pdf}
├── fig3_heatmap.{png,pdf}
├── fig4_scalability.{png,pdf}
├── fig5_gamma.{png,pdf}
├── fig6_tictactoe.{png,pdf}
└── fig7_radar.{png,pdf}
```
