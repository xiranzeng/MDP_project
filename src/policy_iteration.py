import numpy as np

class PolicyIterationAgent:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.n_states)
        self.gamma = env.gamma
        self.policy = np.zeros(env.n_states, dtype=int)
    
    def compute_state_value(self, s, policy, V):
        """计算给定策略下状态 s 的值（贝尔曼期望方程）"""
        a = policy[s]
        value = 0.0
        for next_state in range(self.env.n_states):
            prob = self.env.P[s, a, next_state]
            if prob > 0:
                cost = self.env.C[s, a]
                value += prob * (cost + self.gamma * V[next_state])
        return value
    
    def policy_evaluation(self, theta=1e-6, max_iterations=10000):
        """
        策略评估：迭代计算当前策略的值函数
        """
        V = self.V.copy()
        iteration = 0
        
        while iteration < max_iterations:
            delta = 0
            V_new = V.copy()
            
            for s in range(self.env.n_states):
                # 使用当前策略计算状态值
                V_new[s] = self.compute_state_value(s, self.policy, V)
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
        
        for s in range(self.env.n_states):
            # 计算所有动作的值
            action_values = np.zeros(self.env.n_actions)
            for a in range(self.env.n_actions):
                for next_state in range(self.env.n_states):
                    prob = self.env.P[s, a, next_state]
                    if prob > 0:
                        cost = self.env.C[s, a]
                        action_values[a] += prob * (cost + self.gamma * self.V[next_state])
            
            # 选择最优动作（最小化 cost）
            best_action = np.argmin(action_values)
            
            if best_action != self.policy[s]:
                policy_stable = False
                new_policy[s] = best_action
        
        return new_policy, policy_stable
    
    def optimize(self, theta=1e-4, max_iterations=1000):
        """
        Policy Iteration 主循环
        交替进行策略评估和策略改进，直到策略收敛
        
        返回:
            policy: 最优策略
            V: 最优值函数
        """
        iteration = 0
        
        # 初始化随机策略
        self.policy = np.random.randint(0, self.env.n_actions, size=self.env.n_states)
        
        while iteration < max_iterations:
            # 策略评估
            self.V = self.policy_evaluation(theta=theta)
            
            # 策略改进
            new_policy, policy_stable = self.policy_improvement()
            
            print(f"Iteration {iteration}: policy_stable = {policy_stable}")
            
            if policy_stable:
                print(f"Policy converged after {iteration} iterations")
                break
            
            self.policy = new_policy
            iteration += 1
        
        # 最终策略评估得到最优值函数
        self.V = self.policy_evaluation(theta=theta)
        
        return self.policy.copy(), self.V.copy()