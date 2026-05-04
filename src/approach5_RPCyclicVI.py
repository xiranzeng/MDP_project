import numpy as np


class RPCyclicVIAgent:
    """
    Randomly Permuted Cyclic Value Iteration (Approach 5)
    兼容 MDPAdapter / TicTacToe 环境
    """
    
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        
        # 兼容属性命名：MDPAdapter 使用 nS/nA，原始 MDP 使用 n_states/n_actions
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states
            self.n_actions = env.n_actions
        else:
            self.n_states = env.nS
            self.n_actions = env.nA
        
        self.V = np.zeros(self.n_states)
        self.round_num = 0

    def _get_transitions(self, s, a):
        """
        获取转移概率，自动兼容两种格式：
        - 列表格式：env.P[s][a] = [(prob, next_state, reward, done), ...]
        - NumPy 格式：env.P[s, a, ns]
        """
        # 检查是否有 shape 属性（NumPy 数组）
        if hasattr(self.env.P, 'shape') and len(self.env.P.shape) == 3:
            # NumPy 数组格式
            transitions = []
            for ns in range(self.n_states):
                prob = self.env.P[s, a, ns]
                if prob > 0:
                    # 获取奖励（成本取负）
                    if hasattr(self.env, 'R'):
                        reward = self.env.R[s, a]
                    elif hasattr(self.env, 'C'):
                        reward = -self.env.C[s, a]
                    else:
                        reward = 0.0
                    transitions.append((prob, ns, reward, False))
            return transitions
        else:
            # 列表格式（MDPAdapter）
            return self.env.P[s][a]

    def _get_shape(self):
        """获取环境形状（用于可视化）"""
        if hasattr(self.env, 'shape'):
            return self.env.shape
        elif hasattr(self.env, 'mdp_name'):
            return (1, self.n_states)
        else:
            return (1, self.n_states)

    def next_best_action(self, s, V):
        """计算状态 s 的最优动作和最优值"""
        action_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            for prob, next_state, reward, done in self._get_transitions(s, a):
                action_values[a] += prob * (reward + self.gamma * V[next_state])
        
        best_action = np.argmax(action_values)
        best_value = np.max(action_values)
        return best_action, best_value

    def optimize(self, theta=1e-6, max_iterations=10000, random_seed=None, **kwargs):
        """
        Randomly Permuted Cyclic Value Iteration (Approach 5)
        
        在每次迭代中，使用随机排列的顺序更新所有状态
        每个状态更新后立即使用新值，且每个状态在本轮中只更新一次
        
        参数:
            theta: 收敛阈值
            max_iterations: 最大迭代次数
            random_seed: 随机种子（用于可重复性）
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.V = np.zeros(self.n_states)
        delta = float("inf")
        self.round_num = 0
        
        while delta > theta and self.round_num < max_iterations:
            # 初始化 \tilde{y}^k = y^k
            y_tilde = self.V.copy()
            # 初始化 B^k = {1, 2, ..., m}
            states_left = list(range(self.n_states))
            delta = 0
            
            # 可选：每 50 轮打印一次进度
            if self.round_num % 50 == 0:
                try:
                    shape = self._get_shape()
                    print(f"\nRPCyclicVI: Round {self.round_num}")
                    print(np.reshape(self.V, shape))
                except:
                    print(f"\nRPCyclicVI: Round {self.round_num}, delta={delta:.6f}")
            
            # 随机选择状态，不放回地更新所有状态
            while states_left:
                # 随机从 B^k 中选择一个状态 i
                s = np.random.choice(states_left)
                
                # 使用当前的 y_tilde 计算最优值（已包含本轮已更新状态的值）
                best_action, best_action_value = self.next_best_action(s, y_tilde)
                
                # 记录该状态的变化量
                state_delta = np.abs(best_action_value - y_tilde[s])
                if state_delta > delta:
                    delta = state_delta
                
                # 立即更新 y_tilde 中的值
                y_tilde[s] = best_action_value
                
                # 从 B^k 中移除该状态
                states_left.remove(s)
            
            # y^{k+1} = \tilde{y}^k
            self.V = y_tilde
            self.round_num += 1
        
        print(f"\nRPCyclicVI converged after {self.round_num} rounds, final delta={delta:.6f}")
        
        # 计算最终策略
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            best_action, best_action_value = self.next_best_action(s, self.V)
            policy[s] = best_action
        
        return policy, self.V


# 保持与原代码一致的类名
Agent = RPCyclicVIAgent