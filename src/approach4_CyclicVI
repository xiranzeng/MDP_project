import numpy as np

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

    def optimize(self):  # core method - CyclicVI (Approach 4)
        """
        Cyclic Value Iteration (Approach 4)
        在每次迭代中，按顺序更新状态，并立即使用新更新的值
        """
        THETA = 0.0001
        delta = float("inf")
        round_num = 0

        while delta > THETA:
            # 初始化 \tilde{y}^k = y^k
            y_tilde = self.V.copy()
            delta = 0
            
            print("\nValue Iteration: Round " + str(round_num))
            print(np.reshape(self.V, self.env.shape))
            
            # For i = 1 to m (按顺序更新所有状态)
            for s in range(self.env.nS):
                # 使用当前的 y_tilde 计算最优值（已包含前面状态的更新）
                best_action, best_action_value = self.next_best_action(s, y_tilde)
                # 记录该状态的变化量
                state_delta = np.abs(best_action_value - y_tilde[s])
                if state_delta > delta:
                    delta = state_delta
                # 立即更新 y_tilde 中的值
                y_tilde[s] = best_action_value
            
            # y^{k+1} = \tilde{y}^k
            self.V = y_tilde
            round_num += 1

        # 计算最终策略
        policy = np.zeros(self.env.nS)
        for s in range(self.env.nS):
            best_action, best_action_value = self.next_best_action(s, self.V)
            policy[s] = best_action

        return policy