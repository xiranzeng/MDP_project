# %%
import numpy as np
from Env import GridWorld


DISCOUNT_FACTOR = 1 # 请根据不同的任务使用具体discount_factor


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

    def optimize(self): # core method
        THETA = 0.0001
        delta = float("inf")
        round_num = 0

        while delta > THETA:
            # 循环条件：只要最大变化量大于阈值，就继续迭代,
            # 当所有状态的价值变化都很小时（≤0.0001），停止迭代
            delta = 0
            # 重置本轮的最大变化量
            # 准备记录本轮中各状态的变化
            print("\nValue Iteration: Round " + str(round_num))
            print(np.reshape(self.V, self.env.shape))
            for s in range(self.env.nS):
                best_action, best_action_value = self.next_best_action(s, self.V)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                self.V[s] = best_action_value
            round_num += 1

        policy = np.zeros(self.env.nS)
        for s in range(self.env.nS):
            best_action, best_action_value = self.next_best_action(s, self.V)
            policy[s] = best_action

        return policy


# env = GridWorld()
# agent = Agent(env)
# policy = agent.optimize()
# print("\nBest Policy")
# print(np.reshape([env.get_action_name(entry) for entry in policy], env.shape))

# env = GridWorld(wind_prob=.2)
# agent = Agent(env)
# policy = agent.optimize()
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
policy = agent.optimize()

print("\nFinal Value Function:")
print(agent.V.reshape(env.shape))
print("\nFinal Policy (0=RIGHT,1=LEFT):")
print(policy.reshape(env.shape))

# %%
print("="*60)
print("Testing Gambler's MDP")
print("="*60)

env = MDPAdapter("gambler", goal=50, p_win=0.4, gamma=0.95)
DISCOUNT_FACTOR = 0.95
agent = Agent(env)
policy = agent.optimize()

print("\nValue Function (first 20):", agent.V[:20])
print("Policy (first 20):", policy[:20])

# %%
print("="*60)
print("Testing GridWorld MDP")
print("="*60)

# env = MDPAdapter("gridworld", k=4, gamma=0.9, slip_prob=0.1)
DISCOUNT_FACTOR = 0.9
# agent = Agent(env)
# policy = agent.optimize()


slip_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = {}

for slip_prob in slip_probs:
    env = MDPAdapter("gridworld", k=5, gamma=0.9, slip_prob=slip_prob)
    agent = Agent(env)
    density = (env.P > 0).mean()
    policy = agent.optimize()
    results[f'prob={slip_prob}'] = [(density)]

for slip_prob in slip_probs:
    print(results[slip_prob])

print("\nValue Function:")
print(agent.V.reshape(env.shape))
print("\nPolicy (0=UP,1=DOWN,2=LEFT,3=RIGHT):")
print(policy.reshape(env.shape))


