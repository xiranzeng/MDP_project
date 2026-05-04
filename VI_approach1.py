# %%
import numpy as np


class ValueIterationAgent:
    """标准值迭代算法 - 兼容 mdp_lib 和 TicTacToe MDP 接口"""
    
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        
        # 兼容两种属性命名
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states
            self.n_actions = env.n_actions
        else:
            self.n_states = env.nS
            self.n_actions = env.nA
        
        self.V = np.zeros(self.n_states)
        self.round_num = 0

    def _get_transitions_and_rewards(self, s, a):
        """
        统一获取转移概率和奖励
        返回: list of (prob, next_state, reward)
        """
        # 检查是哪种接口
        if hasattr(self.env.P, 'shape') and len(self.env.P.shape) == 3:
            # 新接口：NumPy 数组格式 (来自 mdp_lib 或 TicTacToe)
            transitions = []
            for next_s in range(self.n_states):
                prob = self.env.P[s, a, next_s]
                if prob > 0:
                    # 获取奖励（成本取负）
                    if hasattr(self.env, 'R'):
                        reward = self.env.R[s, a]
                    elif hasattr(self.env, 'C'):
                        reward = -self.env.C[s, a]
                    else:
                        reward = 0.0
                    transitions.append((prob, next_s, reward))
            return transitions
        else:
            # 旧接口：列表格式 (来自 GridWorld)
            transitions = []
            for prob, next_state, reward, done in self.env.P[s][a]:
                transitions.append((prob, next_state, reward))
            return transitions

    def next_best_action(self, s, V):
        """返回 (best_action, best_value)"""
        action_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            for prob, next_state, reward in self._get_transitions_and_rewards(s, a):
                action_values[a] += prob * (reward + self.gamma * V[next_state])
        
        best_action = np.argmax(action_values)
        best_value = np.max(action_values)
        return best_action, best_value

    def _get_shape(self):
        """获取环境形状（用于可视化）"""
        if hasattr(self.env, 'shape'):
            return self.env.shape
        elif hasattr(self.env, 'mdp_name'):
            return (1, self.n_states)
        else:
            return (1, self.n_states)

    def optimize(self, theta=1e-4, max_iterations=10000):
        """核心优化方法"""
        self.V = np.zeros(self.n_states)
        delta = float("inf")
        round_num = 0

        while delta > theta and round_num < max_iterations:
            delta = 0
            V_new = self.V.copy()
            
            print(f"\nValue Iteration: Round {round_num}")
            # 安全地打印值函数
            try:
                shape = self._get_shape()
                print(np.reshape(self.V, shape))
            except:
                print(self.V[:10])  # 只打印前10个
            
            for s in range(self.n_states):
                best_action, best_action_value = self.next_best_action(s, self.V)
                V_new[s] = best_action_value
                delta = max(delta, np.abs(best_action_value - self.V[s]))
            
            self.V = V_new
            round_num += 1
        
        self.round_num = round_num

        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            best_action, _ = self.next_best_action(s, self.V)
            policy[s] = best_action

        return policy, self.V  # 统一返回两个值


# %%
# 测试代码
if __name__ == "__main__":
    from tictactoe_mdp import tictactoe_mdp
    
    # 创建 TicTacToe MDP
    mdp = tictactoe_mdp(gamma=0.99)
    
    # 添加兼容属性（让 ValueIterationAgent 能识别）
    mdp.nS = mdp.n_states
    mdp.nA = mdp.n_actions
    mdp.shape = (1, mdp.n_states)
    
    # 运行算法
    agent = ValueIterationAgent(mdp, gamma=0.99)
    policy, V = agent.optimize(theta=1e-4)
    
    #print(f"\n收敛迭代次数: {agent.round_num}")
    #print(f"策略形状: {policy.shape}")
    print(f"值函数形状: {V.shape}")