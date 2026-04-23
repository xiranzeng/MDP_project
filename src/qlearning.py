import numpy as np

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, epsilon=0.1):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma
        self.lr = learning_rate  # 学习率 α
        self.epsilon = epsilon    # 探索率 ε
        
        # 初始化 Q 表
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.V = np.zeros(self.n_states)  # 保持接口一致
    
    def epsilon_greedy_policy(self, s):
        """ε-greedy 策略选择动作"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择 Q 值最小的动作（最小化 cost）
            return np.argmin(self.Q[s])
    
    def get_action_value(self, s, a, V=None):
        """计算状态-动作值（如果提供 V 则用 V，否则用 Q）"""
        value = 0.0
        for next_state in range(self.n_states):
            prob = self.env.P[s, a, next_state]
            if prob > 0:
                cost = self.env.C[s, a]
                if V is not None:
                    value += prob * (cost + self.gamma * V[next_state])
                else:
                    # 使用 Q 值中的最小值作为下一状态的值
                    next_value = np.min(self.Q[next_state]) if len(self.Q[next_state]) > 0 else 0
                    value += prob * (cost + self.gamma * next_value)
        return value
    
    def optimize(self, episodes=10000, max_steps_per_episode=1000, theta=1e-4):
        """
        Q-Learning 算法
        
        参数:
            episodes: 训练的回合数
            max_steps_per_episode: 每个回合的最大步数
            theta: 用于判断收敛的阈值（可选，Q-learning 通常用 episodes 控制）
        
        返回:
            policy: 最优策略
            V: 最优值函数（从 Q 表导出）
        """
        # 可选：添加衰减的探索率
        epsilon_start = self.epsilon
        epsilon_end = 0.01
        epsilon_decay = 0.995
        
        print(f"Starting Q-Learning with {episodes} episodes...")
        
        for episode in range(episodes):
            # 衰减探索率
            if episode % 100 == 0 and episode > 0:
                self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)
                if episode % 1000 == 0:
                    print(f"Episode {episode}, epsilon = {self.epsilon:.4f}")
            
            # 从随机状态开始，或者从状态 0 开始
            s = np.random.randint(self.n_states)
            # 或者使用固定起始状态: s = 0
            
            total_cost = 0
            step = 0
            
            while step < max_steps_per_episode:
                # 选择动作
                a = self.epsilon_greedy_policy(s)
                
                # 执行动作，观察下一个状态和 cost
                # 需要根据概率采样下一个状态
                probs = self.env.P[s, a]
                next_state = np.random.choice(self.n_states, p=probs)
                cost = self.env.C[s, a]
                total_cost += cost
                
                # Q-Learning 更新公式
                # Q(s,a) ← Q(s,a) + α [r + γ * min_a' Q(s',a') - Q(s,a)]
                # 注意：这里用的是 cost，所以是加号，但因为是 cost 要最小化，所以用 min
                if next_state < self.n_states:
                    best_next_q = np.min(self.Q[next_state])
                else:
                    best_next_q = 0
                
                td_target = cost + self.gamma * best_next_q
                td_error = td_target - self.Q[s, a]
                self.Q[s, a] += self.lr * td_error
                
                # 转移到下一状态
                s = next_state
                step += 1
                
                # 可选：如果到达终止状态，提前结束
                # 假设最后一个状态是终止状态
                if s == self.n_states - 1:
                    break
            
            # 每 1000 个 episode 打印一次进度
            if (episode + 1) % 1000 == 0:
                avg_cost = total_cost / (step + 1)
                print(f"Episode {episode + 1}/{episodes}, avg_cost = {avg_cost:.4f}")
        
        # 从 Q 表导出值函数和策略
        self.V = np.min(self.Q, axis=1)  # V(s) = min_a Q(s,a)
        policy = np.argmin(self.Q, axis=1)  # π(s) = argmin_a Q(s,a)
        
        print(f"Q-Learning finished. Final epsilon = {self.epsilon:.4f}")
        
        return policy, self.V