# %%
import numpy as np
import random


class RandomVIAgent:
    """Random Value Iteration - 随机值迭代"""
    
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        
        # 兼容属性
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states
            self.n_actions = env.n_actions
        else:
            self.n_states = env.nS
            self.n_actions = env.nA
        
        self.V = np.zeros(self.n_states)
        self.round_num = 0

    def _get_transitions(self, s, a):
        """统一获取转移概率（兼容两种格式）"""
        if hasattr(self.env.P, 'shape') and len(self.env.P.shape) == 3:
            # NumPy 数组格式
            transitions = []
            for ns in range(self.n_states):
                prob = self.env.P[s, a, ns]
                if prob > 0:
                    if hasattr(self.env, 'R'):
                        reward = self.env.R[s, a]
                    else:
                        reward = -self.env.C[s, a]
                    transitions.append((prob, ns, reward, False))
            return transitions
        else:
            # 列表格式
            return self.env.P[s][a]

    def next_best_action(self, s, V):
        action_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            for prob, next_state, reward, done in self._get_transitions(s, a):
                action_values[a] += prob * (reward + self.gamma * V[next_state])
        
        best_action = np.argmax(action_values)
        best_value = np.max(action_values)
        return best_action, best_value

    def optimize(self, subset_ratio=0.3, theta=1e-4, max_iterations=10000, **kwargs):
        """
        Random Value Iteration
        
        参数:
            subset_ratio: 每轮更新状态的比例
            theta: 收敛阈值
            max_iterations: 最大迭代次数
        """
        self.V = np.zeros(self.n_states)
        delta = float("inf")
        self.round_num = 0
        
        subset_size = max(1, int(self.n_states * subset_ratio))
        
        while delta > theta and self.round_num < max_iterations:
            delta = 0
            # 随机选择要更新的状态子集
            Bk = random.sample(range(self.n_states), subset_size)
            
            # 记录本轮更新的状态的最大变化
            for s in Bk:
                _, new_value = self.next_best_action(s, self.V)
                state_delta = abs(new_value - self.V[s])
                if state_delta > delta:
                    delta = state_delta
                self.V[s] = new_value
            
            self.round_num += 1

            
            if self.round_num % 100 == 0:
                print(f"  Round {self.round_num}, delta={delta:.6f}")
        
        print(f"  RandomVI converged after {self.round_num} rounds, final delta={delta:.6f}")
        
        # 提取策略
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            policy[s], _ = self.next_best_action(s, self.V)
        
        return policy, self.V


# %%
# 测试代码
if __name__ == "__main__":
    from mdp_lib import get_mdp
    
    env = get_mdp("chain", n=20, gamma=0.9)
    env.nS = env.n_states
    env.nA = env.n_actions
    
    agent = RandomVIAgent(env, gamma=0.9)
    policy, V = agent.optimize(subset_ratio=0.3, theta=1e-4)
    
    print(f"迭代轮数: {agent.round_num}")
    print(f"策略: {policy[:10]}")