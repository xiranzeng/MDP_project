import numpy as np
from collections import defaultdict
from typing import List, Set, Tuple


class InfluenceTreeAgent:
    """Influence Tree Value Iteration (Approach 3) - 兼容 TicTacToe"""
    
    def __init__(self, env):
        self.env = env
        self.gamma = env.gamma
        
        # 兼容属性命名
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states
            self.n_actions = env.n_actions
        else:
            self.n_states = env.nS
            self.n_actions = env.nA
        
        self.V = np.zeros(self.n_states)
        self.round_num = 0
        self.total_updates = 0
        
        # 构建依赖图
        self._build_dependency_graph()
    
    def _has_numpy_P(self):
        """检查 P 是否是 NumPy 数组格式"""
        return hasattr(self.env.P, 'shape') and len(self.env.P.shape) == 3
    
    def _get_successors(self, s, a) -> Set[int]:
        """获取状态 s 下动作 a 的所有后继状态"""
        successors = set()
        
        if self._has_numpy_P():
            for ns in range(self.n_states):
                if self.env.P[s, a, ns] > 0:
                    successors.add(ns)
        else:
            for prob, next_state, reward, done in self.env.P[s][a]:
                successors.add(next_state)
        
        return successors
    
    def _get_cost(self, s, a) -> float:
        """获取成本"""
        if hasattr(self.env, 'C'):
            if self._has_numpy_P():
                return self.env.C[s, a]
            else:
                return self.env.C[s][a]
        elif hasattr(self.env, 'mdp') and hasattr(self.env.mdp, 'C'):
            return self.env.mdp.C[s, a]
        else:
            return 0.0
    
    def _build_dependency_graph(self):
        n = self.n_states
        
        self.N = [set() for _ in range(n)]
        
        for s in range(n):
            for a in range(self.n_actions):
                successors = self._get_successors(s, a)
                self.N[s].update(successors)
        
        self.P_rev = [set() for _ in range(n)]
        for s in range(n):
            for ns in self.N[s]:
                self.P_rev[ns].add(s)
    
    def get_influence_set(self, B: Set[int]) -> Set[int]:
        I = set()
        for s in B:
            I.update(self.P_rev[s])
        return I
    
    def compute_bellman_value(self, s: int, V: np.ndarray) -> Tuple[int, float]:
        best_value = float('inf')
        best_action = 0
        
        for a in range(self.n_actions):
            cost = self._get_cost(s, a)
            value = cost
            
            if self._has_numpy_P():
                for ns in range(self.n_states):
                    prob = self.env.P[s, a, ns]
                    if prob > 0:
                        value += prob * self.gamma * V[ns]
            else:
                for prob, next_state, reward, done in self.env.P[s][a]:
                    value += prob * self.gamma * V[next_state]
            
            if value < best_value:
                best_value = value
                best_action = a
        
        return best_action, best_value
    
    def compute_bellman_residual(self, s: int, V: np.ndarray) -> float:
        _, tv = self.compute_bellman_value(s, V)
        return abs(tv - V[s])
    
    def optimize(self, theta: float = 1e-6, 
                 max_iterations: int = 10000,
                 residual_threshold_ratio: float = 0.1,
                 verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        
        self.V = np.zeros(self.n_states)
        self.round_num = 0
        self.total_updates = 0
        
        active_set_sizes = []
        B = set(range(self.n_states))
        
        if verbose:
            print(f"Influence Tree Value Iteration started")
            print(f"  States: {self.n_states}, Actions: {self.n_actions}")
        
        for iteration in range(max_iterations):
            V_new = self.V.copy()
            delta = 0
            updated_count = 0
            
            for s in B:
                _, new_value = self.compute_bellman_value(s, self.V)
                V_new[s] = new_value
                state_delta = abs(new_value - self.V[s])
                if state_delta > delta:
                    delta = state_delta
                updated_count += 1
            
            self.total_updates += updated_count
            active_set_sizes.append(len(B))
            self.V = V_new
            self.round_num = iteration + 1
            
            if verbose and (iteration % 10 == 0):
                print(f"  Iter {self.round_num:4d}: |B|={len(B):4d}, delta={delta:.2e}")
            
            if delta < theta:
                if verbose:
                    print(f"  Converged at iteration {self.round_num}")
                break
            
            I = self.get_influence_set(B)
            threshold = theta * residual_threshold_ratio
            B_next = set()
            
            for s in I:
                residual = self.compute_bellman_residual(s, self.V)
                if residual >= threshold:
                    B_next.add(s)
            
            if len(B_next) == 0:
                B_next = I if I else set(range(self.n_states))
            
            B = B_next
        
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            best_action, _ = self.compute_bellman_value(s, self.V)
            policy[s] = best_action
        
        if verbose:
            print(f"  Final: {self.round_num} iterations, total updates = {self.total_updates}")
        
        return policy, self.V