import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable

@dataclass
class MDP:
    n_states:  int
    n_actions: int
    gamma:     float
    P:         np.ndarray   # (n_states, n_actions, n_states)
    C:         np.ndarray   # (n_states, n_actions)


def gamblers_mdp(goal: int = 100, p_win: float = 0.4, gamma: float = 0.9) -> MDP:
    nS = goal + 1
    nA = goal // 2 + 1 
    P = np.zeros((nS, nA, nS))
    C = np.zeros((nS, nA))

    for s in range(nS):
        for a in range(nA):

            if s == 0 or s == goal:
                P[s, a, s] = 1.0
                C[s, a]    = 0.0
                continue

            max_bet = min(s, goal - s)
            if a > max_bet:
                P[s, a, s] = 1.0
                C[s, a]    = 0.0
                continue

            if a == 0:
                P[s, a, s] = 1.0
                C[s, a]    = 0.0
                continue

            win_state  = s + a
            lose_state = s - a

            P[s, a, win_state]  += p_win
            P[s, a, lose_state] += (1 - p_win)
            
            C[s, a] = -p_win if win_state == goal else 0.0

    return MDP(n_states=nS, n_actions=nA, gamma=gamma, P=P, C=C)



UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

def gridworld_mdp(k: int = 5, gamma: float = 0.9, slip_prob: float = 0.0) -> MDP:
    if k < 2:
        raise ValueError("k must be at least 2")
    if not (0 <= slip_prob <= 1):
        raise ValueError("slip_prob must be in [0, 1]")
    if not (0 < gamma < 1):
        raise ValueError("gamma must be in (0, 1)")

    n = k * k
    goal = n - 1

    P = np.zeros((n, 4, n))
    C = np.zeros((n, 4)) 

    # Helper: compute next state given (r, c) and action a
    def next_state(r, c, a):
        dr, dc = MOVES[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < k and 0 <= nc < k:
            return nr * k + nc
        else:
            return r * k + c  # stay in place

    for r in range(k):
        for c in range(k):
            s = r * k + c

            # Goal state: absorbing, zero cost
            if s == goal:
                for a in range(4):
                    P[s, a, goal] = 1.0
                    C[s, a] = 0.0
                continue

            for a in range(4):
                if slip_prob == 0.0:
                    ns = next_state(r, c, a)
                    P[s, a, ns] = 1.0
                    C[s, a] = 0.0 if ns == goal else 1.0
                else:
                    # Stochastic slip: intended direction with prob 1 - slip_prob
                    # Each other direction gets slip_prob / 3
                    probs = {a: 1 - slip_prob}
                    for other in range(4):
                        if other != a:
                            probs[other] = slip_prob / 3

                    for a2, prob in probs.items():
                        ns = next_state(r, c, a2)
                        P[s, a, ns] += prob
                    prob_to_goal = sum(prob for a2, prob in probs.items() if next_state(r, c, a2) == goal)
                    C[s, a] = 1.0 - prob_to_goal


    return MDP(n_states=n, n_actions=4, gamma=gamma, P=P, C=C)


def chain_mdp(n: int = 20, p: float = 0.9, gamma: float = 0.9) -> MDP:
    RIGHT_ACTION, LEFT_ACTION = 0, 1
    P = np.zeros((n, 2, n))
    C = np.zeros((n, 2))

    for i in range(n):
        if i == 0 or i == n - 1:
            P[i, RIGHT_ACTION, i] = 1.0
            P[i, LEFT_ACTION,  i] = 1.0
            C[i, RIGHT_ACTION] = 0.0
            C[i, LEFT_ACTION]  = 0.0
            continue

        next_r = min(i + 1, n - 1)
        P[i, RIGHT_ACTION, next_r] = p
        P[i, RIGHT_ACTION, i] += (1 - p)
        prob_to_goal_right = p if next_r == n - 1 else 0.0
        C[i, RIGHT_ACTION] = 1.0 - prob_to_goal_right

        next_l = max(i - 1, 0)
        P[i, LEFT_ACTION, next_l] = p
        P[i, LEFT_ACTION, i] += (1 - p)
        prob_to_goal_left = p if next_l == 0 else 0.0
        C[i, LEFT_ACTION] = 1.0 - prob_to_goal_left

    return MDP(n_states=n, n_actions=2, gamma=gamma, P=P, C=C)


def random_mdp(n_states: int = 50, n_actions: int = 4, gamma: float = 0.9, density: float = 1.0, seed: int = None) -> MDP:
    if seed is not None:
        np.random.seed(seed)
    
    if not (0 < density <= 1):
        raise ValueError("density must be in (0, 1]")
    
    P = np.zeros((n_states, n_actions, n_states))
    
    for s in range(n_states):
        for a in range(n_actions):
            probs = np.random.rand(n_states)
            
            if density < 1.0:
                mask = np.random.rand(n_states) < density
                probs[~mask] = 0
            
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.ones(n_states) / n_states
            
            P[s, a, :] = probs

    C = np.random.rand(n_states, n_actions)
    
    return MDP(n_states=n_states, n_actions=n_actions, gamma=gamma, P=P, C=C)


MDP_REGISTRY: Dict[str, Callable[..., MDP]] = {
    "chain": chain_mdp,
    "gambler": gamblers_mdp,
    "gridworld": gridworld_mdp,
    "random": random_mdp,
}

def get_mdp(name: str, **kwargs) -> MDP:
    if name not in MDP_REGISTRY:
        raise ValueError(f"Unknown MDP: {name}. Available: {list(MDP_REGISTRY.keys())}")
    return MDP_REGISTRY[name](**kwargs)

if __name__ == "__main__":
    for name in MDP_REGISTRY:
        mdp = get_mdp(name)
        print(f"{name:10} | states={mdp.n_states:3d}, actions={mdp.n_actions:3d}, "
              f"density={(mdp.P > 0).mean():.2%}")
