import numpy as np
from collections import defaultdict
from typing import List, Set, Tuple


class InfluenceTreeAgent:
    """Influence Tree Value Iteration (Approach 3)"""
    
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.n_states)
        self.gamma = env.gamma
        self.round_num = 0
        self.total_updates = 0  # 记录总更新次数（用于效率分析）
        
        # 构建依赖图
        self._build_dependency_graph()
    
    def _build_dependency_graph(self):
        """
        构建依赖图
        
        N[i]: 状态 i 依赖的后继状态集（i 的 Bellman 更新需要哪些状态的值）
        P_rev[s]: 反向依赖，哪些状态的更新依赖于状态 s
        """
        n = self.env.n_states
        
        # N[i] = 状态 i 依赖的后继状态
        self.N = [set() for _ in range(n)]
        
        for s in range(n):
            for a in range(self.env.n_actions):
                for ns in range(n):
                    if self.env.P[s, a, ns] > 0:
                        self.N[s].add(ns)
        
        # P_rev[s] = 依赖于状态 s 的状态集合（反向依赖）
        self.P_rev = [set() for _ in range(n)]
        
        for s in range(n):
            for ns in self.N[s]:
                self.P_rev[ns].add(s)
        
    
    def get_influence_set(self, B: Set[int]) -> Set[int]:
        """
        计算影响集 I(B)
        
        I(B) = ∪_{s∈B} P_rev[s]
        即：如果 B 中的状态值发生变化，I(B) 中的状态可能需要重新计算
        """
        I = set()
        for s in B:
            I.update(self.P_rev[s])
        return I
    
    def compute_bellman_value(self, s: int, V: np.ndarray) -> Tuple[int, float]:
        """
        计算状态 s 的 Bellman 最优值和最优动作
        
        Returns:
            best_action: 最优动作索引
            best_value: 最优值
        """
        best_value = float('inf')
        best_action = 0
        
        for a in range(self.env.n_actions):
            value = self.env.C[s, a]
            for ns in range(self.env.n_states):
                prob = self.env.P[s, a, ns]
                if prob > 0:
                    value += prob * self.gamma * V[ns]
            
            if value < best_value:
                best_value = value
                best_action = a
        
        return best_action, best_value
    
    def compute_bellman_residual(self, s: int, V: np.ndarray) -> float:
        """
        计算状态 s 的 Bellman 残差 |T(V)_s - V_s|
        """
        _, tv = self.compute_bellman_value(s, V)
        return abs(tv - V[s])
    
    def optimize(self, theta: float = 1e-6, 
                 max_iterations: int = 10000,
                 residual_threshold_ratio: float = 0.1,
                 verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Influence Tree Value Iteration 主循环
        
        参数:
            theta: 收敛阈值
            max_iterations: 最大迭代次数
            residual_threshold_ratio: 残差阈值比例，用于筛选下一轮活跃状态
                                      (保留残差 > theta * ratio 的状态)
            verbose: 是否打印详细信息
        
        返回:
            policy: 最优策略
            V: 最优值函数
        """
        self.V = np.zeros(self.env.n_states)
        self.round_num = 0
        self.total_updates = 0
        
        # 记录每轮更新的活跃集大小（用于分析）
        active_set_sizes = []
        
        # 初始活跃集: 所有状态（第一轮全部更新）
        B = set(range(self.env.n_states))
        
        if verbose:
            print(f"Influence Tree Value Iteration started")
            print(f"  States: {self.env.n_states}, Actions: {self.env.n_actions}")
            print(f"  Avg |N(i)|: {np.mean([len(x) for x in self.N]):.2f}")
            print(f"  Avg |P(s)|: {np.mean([len(x) for x in self.P_rev]):.2f}")
        
        for iteration in range(max_iterations):
            V_new = self.V.copy()
            delta = 0
            
            # 只更新活跃集中的状态
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
                print(f"  Iter {self.round_num:4d}: |B|={len(B):4d}, "
                      f"delta={delta:.2e}, updates={self.total_updates}")
            
            # 收敛检查
            if delta < theta:
                if verbose:
                    print(f"  Converged at iteration {self.round_num}")
                break
            
            # 计算下一轮活跃集
            # 1. 计算影响集 I(B)
            I = self.get_influence_set(B)
            
            # 2. 只保留残差大于阈值的状态
            threshold = theta * residual_threshold_ratio
            B_next = set()
            for s in I:
                residual = self.compute_bellman_residual(s, self.V)
                if residual >= threshold:
                    B_next.add(s)
            
            # 确保至少有一个状态被更新（防止提前终止）
            if len(B_next) == 0:
                # 如果没有状态需要更新，但 delta 还没收敛，说明有遗漏
                # 用整个影响集作为备份
                B_next = I if I else set(range(self.env.n_states))
            
            B = B_next
        
        # 提取最优策略
        policy = np.zeros(self.env.n_states, dtype=int)
        for s in range(self.env.n_states):
            best_action, _ = self.compute_bellman_value(s, self.V)
            policy[s] = best_action
        
        if verbose:
            print(f"  Final: {self.round_num} iterations, "
                  f"avg |B| = {np.mean(active_set_sizes):.1f}, "
                  f"total updates = {self.total_updates}")
        
        return policy, self.V


def test_influence_tree():
    """测试 Influence Tree VI"""
    from mdp_lib import chain_mdp, gridworld_mdp, gamblers_mdp, star_mdp
    
    print("="*60)
    print("Testing Influence Tree Value Iteration")
    print("="*60)
    
    # 测试 Chain MDP
    print("\n--- Chain MDP (n=20) ---")
    env = chain_mdp(n=20, gamma=0.9)
    agent = InfluenceTreeAgent(env)
    policy, values = agent.optimize(verbose=True, theta=1e-6)
    
    print(f"  Policy (first 5): {policy[:5]}")
    print(f"  Values (first 5): {values[:5]}")
    print(f"  Iterations: {agent.round_num}")
    print(f"  Total updates: {agent.total_updates} (vs regular VI would be {agent.round_num * env.n_states})")
    
    # 测试 Gridworld
    print("\n--- Gridworld MDP (k=4) ---")
    env = gridworld_mdp(k=4, gamma=0.9, slip_prob=0.1)
    agent = InfluenceTreeAgent(env)
    policy, values = agent.optimize(verbose=True, theta=1e-6)
    
    print(f"  Iterations: {agent.round_num}")
    print(f"  Total updates: {agent.total_updates}")
    
    # 对比: 标准 VI 的更新次数
    from value_iteration import ValueIterationAgent
    agent_vi = ValueIterationAgent(env)
    _, _ = agent_vi.optimize(theta=1e-6)
    vi_updates = agent_vi.round_num * env.n_states
    print(f"  Regular VI would do: {vi_updates} updates")
    print(f"  InfluenceTree saves: {(1 - agent.total_updates/vi_updates)*100:.1f}%")


if __name__ == "__main__":
    test_influence_tree()