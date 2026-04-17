import numpy as np
from typing import Tuple, List, Dict
from collections import deque

class TicTacToeEnv:
    """k×k 井字棋环境 - 从AI视角建模"""
    
    def __init__(self, k: int = 3):
        self.k = k
        self.board_size = k * k
        
        self.AI = 1      # AI (X)
        self.OPPONENT = 2  # 对手 (O)
        self.EMPTY = 0
        
        self._build_state_space()
        
    def _build_state_space(self):
        """构建所有可达状态"""
        self.all_states = []
        self.state_to_idx = {}
        
        empty_board = tuple([self.EMPTY] * self.board_size)
        self.state_to_idx[empty_board] = 0
        self.all_states.append(empty_board)
        
        queue = deque([empty_board])
        
        while queue:
            board = queue.popleft()
            
            if self._is_terminal(board):
                continue
                
            current_player = self._get_current_player(board)
            
            for pos in range(self.board_size):
                if board[pos] == self.EMPTY:
                    new_board_list = list(board)
                    new_board_list[pos] = current_player
                    new_board = tuple(new_board_list)
                    
                    if new_board not in self.state_to_idx:
                        self.state_to_idx[new_board] = len(self.all_states)
                        self.all_states.append(new_board)
                        queue.append(new_board)
        
        self.n_states = len(self.all_states)
        self.n_actions = self.board_size
        
        # 构建转移概率和奖励
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))  # 奖励，不是成本
        
        self._build_transitions()
    
    def _get_current_player(self, board):
        """判断当前轮到谁"""
        count_ai = board.count(self.AI)
        count_opp = board.count(self.OPPONENT)
        
        if count_ai == count_opp:
            return self.AI
        else:
            return self.OPPONENT
    
    def _check_win(self, board, player):
        """检查指定玩家是否获胜"""
        # 行
        for i in range(self.k):
            if all(board[i * self.k + j] == player for j in range(self.k)):
                return True
        
        # 列
        for j in range(self.k):
            if all(board[i * self.k + j] == player for i in range(self.k)):
                return True
        
        # 主对角线
        if all(board[i * self.k + i] == player for i in range(self.k)):
            return True
        
        # 副对角线
        if all(board[i * self.k + (self.k - 1 - i)] == player for i in range(self.k)):
            return True
        
        return False
    
    def _is_terminal(self, board):
        """检查是否为终局状态"""
        if self._check_win(board, self.AI) or self._check_win(board, self.OPPONENT):
            return True
        if all(cell != self.EMPTY for cell in board):
            return True
        return False
    
    def _get_reward(self, board):
        """从AI视角获取终局奖励"""
        if self._check_win(board, self.AI):
            return 1.0      # AI赢了
        elif self._check_win(board, self.OPPONENT):
            return -1.0     # AI输了
        else:
            return 0.0      # 平局
    
    def _build_transitions(self):
        """构建转移概率矩阵"""
        for s_idx, board in enumerate(self.all_states):
            # 终局状态：吸收态，奖励0
            if self._is_terminal(board):
                for a in range(self.n_actions):
                    self.P[s_idx, a, s_idx] = 1.0
                    self.R[s_idx, a] = self._get_reward(board)
                continue
            
            current_player = self._get_current_player(board)
            
            for a in range(self.n_actions):
                # 无效动作（位置已有棋子）：留在原地，负奖励惩罚
                if board[a] != self.EMPTY:
                    self.P[s_idx, a, s_idx] = 1.0
                    self.R[s_idx, a] = -0.1  # 惩罚无效动作
                    continue
                
                if current_player == self.AI:
                    # AI回合：确定性转移
                    new_board_list = list(board)
                    new_board_list[a] = self.AI
                    new_board = tuple(new_board_list)
                    next_idx = self.state_to_idx[new_board]
                    self.P[s_idx, a, next_idx] = 1.0
                    
                    # 如果新状态是终局，奖励就是终局奖励；否则为0（延迟奖励）
                    if self._is_terminal(new_board):
                        self.R[s_idx, a] = self._get_reward(new_board)
                    else:
                        self.R[s_idx, a] = 0.0
                
                else:
                    # 对手回合：对手均匀随机选择空位
                    empty_positions = [pos for pos in range(self.board_size) if board[pos] == self.EMPTY]
                    prob_each = 1.0 / len(empty_positions)
                    
                    for pos in empty_positions:
                        new_board_list = list(board)
                        new_board_list[pos] = self.OPPONENT
                        new_board = tuple(new_board_list)
                        next_idx = self.state_to_idx[new_board]
                        self.P[s_idx, a, next_idx] += prob_each
                    
                    # 对手回合的即时奖励为0，奖励在终局时才给
                    self.R[s_idx, a] = 0.0
    
    def render(self, board):
        """打印棋盘"""
        symbols = {self.EMPTY: '.', self.AI: 'X', self.OPPONENT: 'O'}
        for i in range(self.k):
            row = [symbols[board[i * self.k + j]] for j in range(self.k)]
            print(' '.join(row))
        print()


class StandardValueIteration:
    """标准值迭代 - 最大化奖励"""
    
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.n_states)
        self.history = []
    
    def _compute_best_value(self, s, V):
        """计算状态s的最优值（最大化奖励）"""
        best_value = float('-inf')
        best_action = 0
        
        for a in range(self.env.n_actions):
            value = 0.0
            for next_s in range(self.env.n_states):
                prob = self.env.P[s, a, next_s]
                if prob > 0:
                    value += prob * (self.env.R[s, a] + self.gamma * V[next_s])
            
            if value > best_value:
                best_value = value
                best_action = a
        
        return best_action, best_value
    
    def solve(self, max_iter=10000, verbose=True):
        """求解最优策略"""
        iteration = 0
        self.history = []
        
        if verbose:
            print("=" * 60)
            print("标准值迭代求解井字棋")
            print("=" * 60)
            print(f"状态数: {self.env.n_states}")
            print(f"动作数: {self.env.n_actions}")
            print(f"折扣因子 γ: {self.gamma}")
            print(f"收敛阈值 θ: {self.theta}")
            print()
        
        while iteration < max_iter:
            V_new = np.zeros(self.env.n_states)
            delta = 0
            
            for s in range(self.env.n_states):
                _, best_value = self._compute_best_value(s, self.V)
                V_new[s] = best_value
                delta = max(delta, abs(best_value - self.V[s]))
            
            self.history.append(delta)
            self.V = V_new
            
            if verbose and (iteration % 10 == 0 or delta < self.theta):
                print(f"迭代 {iteration:4d}: delta = {delta:.8f}")
            
            if delta < self.theta:
                if verbose:
                    print(f"\n✓ 收敛于第 {iteration} 次迭代")
                break
            
            iteration += 1
        
        # 提取策略
        policy = np.zeros(self.env.n_states, dtype=int)
        for s in range(self.env.n_states):
            policy[s], _ = self._compute_best_value(s, self.V)
        
        return policy, self.V, iteration, self.history


def evaluate_policy(env, policy, n_games=100, verbose=True):
    """评估策略"""
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(n_games):
        board = tuple([env.EMPTY] * env.board_size)
        
        while not env._is_terminal(board):
            current_player = env._get_current_player(board)
            state_idx = env.state_to_idx[board]
            
            if current_player == env.AI:
                action = policy[state_idx]
            else:
                empty = [p for p in range(env.board_size) if board[p] == env.EMPTY]
                if not empty:
                    break
                action = np.random.choice(empty)
            
            new_board = list(board)
            new_board[action] = current_player
            board = tuple(new_board)
        
        if env._check_win(board, env.AI):
            wins += 1
        elif env._check_win(board, env.OPPONENT):
            losses += 1
        else:
            draws += 1
    
    if verbose:
        print(f"\n策略评估 ({n_games} 局)")
        print(f"  胜: {wins} ({wins/n_games*100:.1f}%)")
        print(f"  负: {losses} ({losses/n_games*100:.1f}%)")
        print(f"  平: {draws} ({draws/n_games*100:.1f}%)")
    
    return wins, losses, draws


def play_game(env, policy):
    """演示一局"""
    board = tuple([env.EMPTY] * env.board_size)
    
    print("\n新游戏：AI (X) vs 随机对手 (O)")
    env.render(board)
    
    while not env._is_terminal(board):
        current_player = env._get_current_player(board)
        state_idx = env.state_to_idx[board]
        
        if current_player == env.AI:
            action = policy[state_idx]
            print(f"AI 下在位置 {action}")
        else:
            empty = [p for p in range(env.board_size) if board[p] == env.EMPTY]
            action = np.random.choice(empty)
            print(f"对手 下在位置 {action}")
        
        new_board = list(board)
        new_board[action] = current_player
        board = tuple(new_board)
        env.render(board)
    
    if env._check_win(board, env.AI):
        print("🎉 AI 获胜！")
    elif env._check_win(board, env.OPPONENT):
        print("😢 AI 输了")
    else:
        print("🤝 平局")


if __name__ == "__main__":
    env = TicTacToeEnv(k=3)
    solver = StandardValueIteration(env, gamma=0.99, theta=1e-6)
    policy, V, iterations, history = solver.solve(verbose=True)
    
    evaluate_policy(env, policy, n_games=500)
    play_game(env, policy)