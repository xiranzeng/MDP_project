import numpy as np


class QLearningAgent:
    """Q-Learning - 兼容 MDPAdapter / TicTacToe"""
    
    def __init__(self, env, learning_rate=0.1, epsilon=0.1):
        self.env = env
        self.gamma = env.gamma
        self.lr = learning_rate
        self.epsilon = epsilon
        
        # 兼容属性命名
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states
            self.n_actions = env.n_actions
        else:
            self.n_states = env.nS
            self.n_actions = env.nA
        
        # 初始化 Q 表
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.V = np.zeros(self.n_states)
        self.round_num = 0

    def _get_cost(self, s, a):
        """获取成本 - 兼容 MDPAdapter"""
        if hasattr(self.env, 'C'):
            if hasattr(self.env.C, 'shape'):
                return self.env.C[s, a]
            else:
                return self.env.C[s][a]
        elif hasattr(self.env, 'mdp') and hasattr(self.env.mdp, 'C'):
            return self.env.mdp.C[s, a]
        else:
            return 0.0

    def _sample_next_state(self, s, a):
        """
        根据转移概率采样下一个状态
        兼容两种 P 格式：
        - NumPy 数组：env.P[s, a, ns]
        - 列表格式：env.P[s][a] = [(prob, next_state, reward, done), ...]
        """
        # 检查 P 的格式
        if hasattr(self.env.P, 'shape') and len(self.env.P.shape) == 3:
            # NumPy 数组格式
            probs = self.env.P[s, a]
            next_state = np.random.choice(self.n_states, p=probs)
        else:
            # 列表格式（MDPAdapter）
            transitions = self.env.P[s][a]
            probs = [t[0] for t in transitions]
            next_states = [t[1] for t in transitions]
            next_state = np.random.choice(next_states, p=probs)
        
        return next_state

    def epsilon_greedy_policy(self, s):
        """ε-greedy 策略选择动作（最小化 cost）"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择 Q 值最小的动作
            return np.argmin(self.Q[s])

    def optimize(self, episodes=10000, max_steps_per_episode=100, verbose=True):
        """
        Q-Learning 算法
        
        参数:
            episodes: 训练的回合数
            max_steps_per_episode: 每个回合的最大步数（井字棋最多9步）
            verbose: 是否打印进度
        
        返回:
            policy: 最优策略
            V: 最优值函数（从 Q 表导出）
        """
        epsilon_start = self.epsilon
        epsilon_end = 0.01
        epsilon_decay = 0.995
        
        if verbose:
            print(f"Starting Q-Learning with {episodes} episodes...")
        
        for episode in range(episodes):
            # 衰减探索率
            if episode % 100 == 0 and episode > 0:
                self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)
                if verbose and episode % 1000 == 0:
                    print(f"  Episode {episode}, epsilon = {self.epsilon:.4f}")
            
            # 从随机状态开始
            s = np.random.randint(self.n_states)
            step = 0
            total_cost = 0
            
            while step < max_steps_per_episode:
                # 选择动作
                a = self.epsilon_greedy_policy(s)
                
                # 执行动作，观察下一个状态和 cost
                next_s = self._sample_next_state(s, a)
                cost = self._get_cost(s, a)
                total_cost += cost
                
                # Q-Learning 更新公式
                best_next_q = np.min(self.Q[next_s])
                td_target = cost + self.gamma * best_next_q
                td_error = td_target - self.Q[s, a]
                self.Q[s, a] += self.lr * td_error
                
                # 转移到下一状态
                s = next_s
                step += 1
                
                # 检查是否到达终局（可选加速）
                # 如果下一状态的 Q 值全部相等，可能是终局
                if np.all(self.Q[next_s] == self.Q[next_s][0]):
                    break
            
            # 每 1000 个 episode 打印一次进度
            if verbose and (episode + 1) % 1000 == 0:
                avg_cost = total_cost / (step + 1) if step > 0 else 0
                print(f"  Episode {episode + 1}/{episodes}, avg_cost = {avg_cost:.4f}")
        
        # 从 Q 表导出值函数和策略
        self.V = np.min(self.Q, axis=1)
        policy = np.argmin(self.Q, axis=1)
        
        if verbose:
            print(f"  Q-Learning finished. Final epsilon = {self.epsilon:.4f}")
        
        self.round_num = episodes
        return policy, self.V