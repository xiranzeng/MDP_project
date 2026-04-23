# %%
import numpy as np
from Env import GridWorld
import random

DISCOUNT_FACTOR = 1  # 请根据不同的任务使用具体discount_factor


class Agent:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.nS)

    def next_best_action(self, s, V):
        action_values = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                action_values[a] += prob * (reward + DISCOUNT_FACTOR * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize_random(self, subset_ratio=0.3):
        """
        RandomVI: 随机值迭代算法
        参数:
        - subset_ratio: 每次更新状态的比例（默认0.3）
        """
        THETA = 0.0001
        delta = float("inf")
        round_num = 0
        
        subset_size = max(1, int(self.env.nS * subset_ratio))
        
        while delta > THETA:
            delta = 0
            round_num += 1
            
            # 随机选择要更新的状态子集 Bk
            Bk = random.sample(range(self.env.nS), subset_size)
            
            # 只更新选中的状态
            for s in Bk:
                _, best_action_value = self.next_best_action(s, self.V)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                self.V[s] = best_action_value
        
        # 提取策略
        policy = np.zeros(self.env.nS)
        for s in range(self.env.nS):
            best_action, _ = self.next_best_action(s, self.V)
            policy[s] = best_action
        
        return policy

# 以下是测试代码
# env = GridWorld()
# agent = Agent(env)
# policy = agent.optimize_random(subset_ratio=0.3)
# print("\nBest Policy")
# print(np.reshape([env.get_action_name(entry) for entry in policy], env.shape))

# %%
from mdp_lib import get_mdp  # 导入同学的MDP

class MDPAdapter:
    """将同学的MDP适配成你的Env格式"""
    
    def __init__(self, mdp_name, **kwargs):
        self.mdp = get_mdp(mdp_name, **kwargs)
        self.nS = self.mdp.n_states
        self.nA = self.mdp.n_actions
        
        # 设置shape用于可视化
        if mdp_name == "gridworld":
            self.shape = (kwargs.get('k', 5), kwargs.get('k', 5))
        elif mdp_name == "chain":
            self.shape = (1, kwargs.get('n', 20))
        elif mdp_name == "gambler":
            goal = kwargs.get('goal', 100)
            self.shape = (1, goal + 1)
        else:
            self.shape = (1, self.nS)
        
        # 构建兼容的P表
        self.P = self._build_transitions()
    
    def _build_transitions(self):
        P = []
        for s in range(self.nS):
            actions = []
            for a in range(self.nA):
                transitions = []
                for next_s in range(self.nS):
                    prob = self.mdp.P[s, a, next_s]
                    if prob > 0:
                        reward = -self.mdp.C[s, a]  # 代价转奖励
                        done = (next_s == self.nS - 1)  # 简化判断
                        transitions.append((prob, next_s, reward, done))
                if len(transitions) == 0:
                    transitions.append((1.0, s, 0, False))
                actions.append(transitions)
            P.append(actions)
        return P
    
    def get_action_name(self, a):
        return f"Action_{a}"

# %%
# 测试1: Chain MDP
print("="*60)
print("Testing Chain MDP")
print("="*60)

env = MDPAdapter("chain", n=50, p=0.9, gamma=0.9)
DISCOUNT_FACTOR = 0.9  # 重要：覆盖原来的1
agent = Agent(env)
policy = agent.optimize_random()

print("\nFinal Value Function:")
print(agent.V.reshape(env.shape))
print("\nFinal Policy (0=RIGHT,1=LEFT):")
print(policy.reshape(env.shape))


