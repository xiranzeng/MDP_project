import numpy as np


class CyclicVIAgent:
    """Cyclic Value Iteration (Approach 4) - 兼容 MDPAdapter / TicTacToe"""
    
    def __init__(self, env):
        self.env = env
        self.gamma = env.gamma
        
        # 兼容属性命名：MDPAdapter 使用 nS/nA，原始 MDP 使用 n_states/n_actions
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states
            self.n_actions = env.n_actions
        else:
            self.n_states = env.nS
            self.n_actions = env.nA
        
        self.V = np.zeros(self.n_states)
        self.round_num = 0

    def _get_cost(self, s, a):
        """获取成本 - 兼容 MDPAdapter（成本在 mdp.C 中）"""
        # 尝试多种可能的存储位置
        if hasattr(self.env, 'C'):
            # 直接有 C 属性
            if hasattr(self.env.C, 'shape'):
                return self.env.C[s, a]
            else:
                return self.env.C[s][a]
        elif hasattr(self.env, 'mdp') and hasattr(self.env.mdp, 'C'):
            # MDPAdapter 包装的情况
            return self.env.mdp.C[s, a]
        else:
            return 0.0

    def _get_value_from_transition(self, s, a, V):
        """计算状态-动作值，自动兼容 P 的两种格式"""
        cost = self._get_cost(s, a)
        value = cost
        
        # 判断 P 的格式
        if hasattr(self.env.P, 'shape') and len(self.env.P.shape) == 3:
            # NumPy 数组格式（P[s, a, ns]）
            for ns in range(self.n_states):
                prob = self.env.P[s, a, ns]
                if prob > 0:
                    value += prob * self.gamma * V[ns]
        else:
            # 列表格式（MDPAdapter: P[s][a] = [(prob, next_state, reward, done), ...]）
            for prob, next_state, reward, done in self.env.P[s][a]:
                value += prob * self.gamma * V[next_state]
        
        return value

    def optimize(self, theta=1e-6, max_iterations=10000, **kwargs):
        """
        Cyclic Value Iteration
        
        参数:
            theta: 收敛阈值
            max_iterations: 最大迭代次数
        """
        self.V = np.zeros(self.n_states)
        delta = float("inf")
        self.round_num = 0
        
        while delta > theta and self.round_num < max_iterations:
            y_tilde = self.V.copy()
            delta = 0
            
            # 按顺序更新所有状态
            for s in range(self.n_states):
                best_value = float('inf')
                
                # 遍历所有动作，找到最优值
                for a in range(self.n_actions):
                    value = self._get_value_from_transition(s, a, y_tilde)
                    if value < best_value:
                        best_value = value
                
                # 记录变化量
                state_delta = abs(best_value - y_tilde[s])
                if state_delta > delta:
                    delta = state_delta
                
                # 立即更新（CyclicVI 的核心）
                y_tilde[s] = best_value
            
            self.V = y_tilde
            self.round_num += 1
            
            # 可选：打印进度
            if self.round_num % 100 == 0:
                print(f"  CyclicVI: Round {self.round_num}, delta={delta:.6f}")
        
        print(f"  CyclicVI converged after {self.round_num} rounds, final delta={delta:.6f}")
        
        # 提取最优策略
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            best_value = float('inf')
            best_action = 0
            for a in range(self.n_actions):
                value = self._get_value_from_transition(s, a, self.V)
                if value < best_value:
                    best_value = value
                    best_action = a
            policy[s] = best_action
        
        return policy, self.V


# 如果你希望保持类名为 Agent（与你的命名一致）
Agent = CyclicVIAgent