# Result Discussion: Seven Methods for Solving MDPs

This folder collects the results for comparing seven MDP solution methods:
`VI`, `RandomVI`, `InfluenceTreeVI`, `CyclicVI`, `RPCyclicVI`, `PolicyIter`, and `Q-Learning`.
The experiments are based on four MDP examples from `example.py`: `chain`, `gambler`, `gridworld`, and `random`.

## 1. What are the best approaches to solve the MDP?

The answer depends on whether the transition model is known and what metric is emphasized.

When the full MDP model is available, the best practical choices in this study are the model-based dynamic programming methods, especially `CyclicVI`, `RPCyclicVI`, `VI`, and `PolicyIter`. On the first three benchmark families (`chain`, `gambler`, `gridworld`), all of these methods reached essentially machine-precision final errors, around `1e-11` to `1e-10`, within the fixed iteration budget. On the more difficult `random` MDP, `VI`, `CyclicVI`, `RPCyclicVI`, `InfluenceTreeVI`, and `PolicyIter` still reached about `8.4e-10`, which is effectively exact for this experiment.

`RandomVI` is the fastest method in wall-clock time on all four examples, but this speed comes from updating only part of the state space in each iteration. As a result, it is not always the most reliable in final accuracy. In particular, on the `random` MDP it stopped at about `7.3e-05`, noticeably worse than the full-sweep methods. Therefore, `RandomVI` is attractive when cheap iterations are important, but it is weaker when strong final accuracy is required.

`Q-Learning` is clearly the weakest method in this comparison. Its final errors remain much larger: about `7.4e-02` on `chain`, `8.6e-02` on `gambler`, `3.8e-01` on `gridworld`, and `5.8e-02` on `random`, while also taking far more time than the planning methods. This is not surprising, because `Q-Learning` is a sample-based reinforcement learning method and does not exploit the known transition model directly. Under the current training budget, it is not competitive with exact planning methods.

Overall, the best conclusion from these experiments is:

- If the transition model is known, use a model-based planner.
- If both speed and accuracy matter, `CyclicVI`, `RPCyclicVI`, and `PolicyIter` are the strongest overall choices.
- If the goal is the lowest wall-clock time on small problems, `RandomVI` is often fastest, but may sacrifice final accuracy.
- `Q-Learning` is not the preferred approach for these fully known MDP benchmarks.

## 2. How do density and sparsity of the transition matrix affect convergence?

The effect of transition structure is visible in both the convergence figures and the parameter sensitivity figure.

For the structured examples (`chain`, `gambler`, and `gridworld`), the transition dynamics are relatively regular and sparse. In these cases, nearly all model-based methods converge to the same near-exact solution. The differences are mostly in update style and overhead, not in eventual accuracy.

For the `random` MDP, the transition matrix is less structured and effectively harder. This is the case where method differences become clearer. `RandomVI` performs worse here than in the other three problems, ending at `7.3e-05` instead of machine precision. The full-sweep methods remain accurate, which suggests that dense or unstructured transitions penalize partial-update methods more than full Bellman sweeps.

From an algorithmic perspective, this makes sense:

- `VI` performs a complete Bellman backup over all states and is therefore robust to density, though each iteration can be more expensive.
- `CyclicVI` and `RPCyclicVI` reuse freshly updated values inside a sweep, so they preserve strong convergence while often improving practical speed.
- `InfluenceTreeVI` is designed to benefit from sparse reachability structure, but in these relatively small benchmarks its bookkeeping overhead offsets much of that theoretical advantage.
- `RandomVI` benefits when a partial update is enough to propagate useful value information, but it becomes less reliable when the transition structure is denser or less local.

The sensitivity plot confirms this pattern. When the `random` MDP density varies from `0.10` to `0.80`, the full-sweep methods remain near `1e-9`, while `RandomVI` stays around `1e-4` to `1e-5` and `Q-Learning` remains around `5e-02` to `7e-02`. In contrast, for `chain`, `gambler`, and `gridworld`, the model-based planners stay essentially exact across the tested parameter ranges.

## 3. What are the main differences among the methods?

### Algorithm design

- `VI` is the standard synchronous Bellman iteration and provides the cleanest baseline.
- `RandomVI` updates only a random subset of states each iteration, reducing per-iteration cost.
- `InfluenceTreeVI` updates states influenced by the current active set, aiming to exploit sparse transition structure.
- `CyclicVI` performs Gauss-Seidel style updates in a fixed order and immediately reuses newly computed values.
- `RPCyclicVI` is the same idea as `CyclicVI`, but with a random permutation each sweep.
- `PolicyIter` alternates policy evaluation and policy improvement, rather than iterating directly on the value function.
- `Q-Learning` learns action values from sampled transitions instead of solving Bellman equations with the known model.

### Theoretical analysis

`VI` is a contraction-mapping method under discounted MDPs, so convergence to the optimal value function is guaranteed. `CyclicVI` and `RPCyclicVI` share the same fixed point but typically improve practical convergence because Gauss-Seidel style reuse of fresh values accelerates information propagation. `PolicyIter` usually requires fewer outer iterations, but each outer step is more expensive because it includes policy evaluation. `RandomVI` and `InfluenceTreeVI` can reduce work per iteration, but their benefit depends more heavily on problem structure. `Q-Learning` converges only asymptotically under suitable stochastic approximation conditions and usually needs far more samples.

### Computation time

The timing results show a clear separation:

- `RandomVI` is the fastest wall-clock method on all four examples.
- `VI`, `CyclicVI`, `RPCyclicVI`, `InfluenceTreeVI`, and `PolicyIter` all finish within roughly `0.02s` to `0.29s` on these benchmarks.
- `Q-Learning` is much slower, taking about `2.15s` to `4.33s`.

So the main time tradeoff is between cheap partial updates (`RandomVI`) and stronger final accuracy (the full-sweep planners).

### Approximation error

Final approximation error is where the strongest difference appears.

- On `chain`, `gambler`, and `gridworld`, all model-based planners reach near-zero error.
- On `random`, all full-sweep planners still reach about `8.4e-10`.
- `RandomVI` degrades on `random` to about `7.3e-05`.
- `Q-Learning` remains one to several orders of magnitude worse than every planning method on all four examples.

This indicates that exact model-based planning is the most reliable strategy for these tasks.

### Sensitivity to model parameters

The parameter sensitivity results show that the model-based planners are very stable across the tested ranges:

- `chain`: varying transition success probability `p`
- `gambler`: varying winning probability `p_win`
- `gridworld`: varying slip probability `slip_prob`
- `random`: varying transition density `density`

For the first three examples, `VI`, `CyclicVI`, `RPCyclicVI`, `InfluenceTreeVI`, and `PolicyIter` remain essentially exact across all tested parameter values. `Q-Learning` changes much more noticeably, especially when `slip_prob` increases in `gridworld`, where its final error rises from almost zero to about `3.1e-01`. On the `random` MDP, the full-sweep planners remain stable, while `RandomVI` and `Q-Learning` show persistently larger errors.

## 4. Final conclusion

The experiments consistently support the same conclusion: when the MDP model is known, model-based dynamic programming methods are the best approaches. Among them, `CyclicVI`, `RPCyclicVI`, and `PolicyIter` offer the strongest overall balance of theory, accuracy, and practical runtime, while `VI` remains the most standard and interpretable baseline. `RandomVI` is appealing for fast approximate updates but is less reliable on harder unstructured problems. `Q-Learning` is not competitive in this setting because it pays the cost of sampling without gaining any advantage from model-free flexibility.

The structure of the transition matrix matters. Sparse and structured problems reduce the gap among exact planners, while denser and less structured transitions expose the weakness of partial-update and sample-based methods. Therefore, the most defensible recommendation from this study is to prefer exact model-based planners whenever the transition model is available, and to treat partial-update or model-free methods as secondary choices driven by computational or modeling constraints rather than by solution quality.

## 5. Limitations

Two limitations should be stated clearly in the report.

- All benchmarks are relatively small, so asymptotic scalability is not fully tested here.
- `Q-Learning` was given the same fixed outer iteration budget as the planning methods, which is convenient for comparison but not necessarily optimal for reinforcement learning.

These limitations do not change the main conclusion, but they help frame the results more carefully.
