import numpy as np


class PolicyIterationAgent:
    """Policy Iteration - 兼容 MDPAdapter / TicTacToe"""
    
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
        self.policy = np.zeros(self.n_states, dtype=int)
        self.round_num = 0

    def _get_cost(self, s, a):
        """获取成本 - 兼容 MDPAdapter（成本在 mdp.C 中）"""
        if hasattr(self.env, 'C'):
            if hasattr(self.env.C, 'shape'):
                return self.env.C[s, a]
            else:
                return self.env.C[s][a]
        elif hasattr(self.env, 'mdp') and hasattr(self.env.mdp, 'C'):
            return self.env.mdp.C[s, a]
        else:
            return 0.0

    def _get_transition_value(self, s, a, V, use_policy=False):
        """
        计算状态-动作值或策略状态值
        
        参数:
            s: 状态
            a: 动作（如果 use_policy=True，则忽略此参数，使用 self.policy[s]）
            V: 值函数
            use_policy: 是否使用当前策略（用于策略评估）
        """
        if use_policy:
            a = self.policy[s]
        
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

    def compute_state_value(self, s, V):
        """计算给定策略下状态 s 的值（贝尔曼期望方程）"""
        return self._get_transition_value(s, None, V, use_policy=True)
    
    def policy_evaluation(self, theta=1e-6, max_iterations=10000):
        """
        策略评估：迭代计算当前策略的值函数
        """
        V = self.V.copy()
        iteration = 0
        
        while iteration < max_iterations:
            delta = 0
            V_new = V.copy()
            
            for s in range(self.n_states):
                V_new[s] = self.compute_state_value(s, V)
                delta = max(delta, np.abs(V_new[s] - V[s]))
            
            V = V_new
            iteration += 1
            
            if delta < theta:
                break
        
        return V
    
    def policy_improvement(self):
        """
        策略改进：对每个状态选择最优动作
        返回策略是否稳定
        """
        policy_stable = True
        new_policy = self.policy.copy()
        
        for s in range(self.n_states):
            # 计算所有动作的值
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                action_values[a] = self._get_transition_value(s, a, self.V, use_policy=False)
            
            # 选择最优动作（最小化 cost）
            best_action = np.argmin(action_values)
            
            if best_action != self.policy[s]:
                policy_stable = False
                new_policy[s] = best_action
        
        return new_policy, policy_stable
    
    def optimize(self, theta=1e-4, max_iterations=1000, verbose=True):
        """
        Policy Iteration 主循环
        交替进行策略评估和策略改进，直到策略收敛
        
        参数:
            theta: 策略评估收敛阈值
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息
        
        返回:
            policy: 最优策略
            V: 最优值函数
        """
        iteration = 0
        
        # 初始化随机策略
        self.policy = np.random.randint(0, self.n_actions, size=self.n_states)
        
        if verbose:
            print(f"Policy Iteration started")
            print(f"  States: {self.n_states}, Actions: {self.n_actions}")
        
        while iteration < max_iterations:
            # 策略评估
            self.V = self.policy_evaluation(theta=theta)
            
            # 策略改进
            new_policy, policy_stable = self.policy_improvement()
            
            if verbose:
                print(f"  Iteration {iteration}: policy_stable = {policy_stable}")
            
            if policy_stable:
                if verbose:
                    print(f"  Policy converged after {iteration} iterations")
                break
            
            self.policy = new_policy
            iteration += 1
        
        self.round_num = iteration
        
        # 最终策略评估得到最优值函数
        self.V = self.policy_evaluation(theta=theta)
        
        if verbose:
            print(f"  Final: {self.round_num} iterations")
        
        return self.policy.copy(), self.V.copy()