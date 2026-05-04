import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dataclasses import dataclass
from typing import Tuple, List, Dict
from collections import deque, defaultdict

from VI_approach1 import ValueIterationAgent
from VI_approach2 import RandomVIAgent
from InfluenceTree import InfluenceTreeAgent
from approach4_CyclicVI import Agent as CyclicVIAgent
from approach5_RPCyclicVI import Agent as RPCyclicVIAgent
from policy_iteration import PolicyIterationAgent
from qlearning import QLearningAgent


@dataclass
class MDP:
    n_states: int
    n_actions: int
    gamma: float
    P: np.ndarray   # (n_states, n_actions, n_states)
    C: np.ndarray   # (n_states, n_actions) 成本（奖励的负值）


class TicTacToeEnv:
    """3x3 井字棋环境，对手为均匀随机"""
    
    def __init__(self):
        self.k = 3
        self.board_size = 9
        
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
        
        # 构建转移概率和成本
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))  # 奖励
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
            # 终局状态：吸收态
            if self._is_terminal(board):
                for a in range(self.n_actions):
                    self.P[s_idx, a, s_idx] = 1.0
                    self.R[s_idx, a] = self._get_reward(board)
                continue
            
            current_player = self._get_current_player(board)
            
            for a in range(self.n_actions):
                # 无效动作
                if board[a] != self.EMPTY:
                    self.P[s_idx, a, s_idx] = 1.0
                    self.R[s_idx, a] = 0.0
                    continue
                
                if current_player == self.AI:
                    # AI回合：确定性转移
                    new_board_list = list(board)
                    new_board_list[a] = self.AI
                    new_board = tuple(new_board_list)
                    next_idx = self.state_to_idx[new_board]
                    self.P[s_idx, a, next_idx] = 1.0
                    
                    if self._is_terminal(new_board):
                        self.R[s_idx, a] = self._get_reward(new_board)
                    else:
                        self.R[s_idx, a] = 0.0
                
                else:
                    # 对手回合：均匀随机选择空位
                    empty_positions = [p for p in range(self.board_size) if board[p] == self.EMPTY]
                    prob_each = 1.0 / len(empty_positions)
                    
                    for pos in empty_positions:
                        new_board_list = list(board)
                        new_board_list[pos] = self.OPPONENT
                        new_board = tuple(new_board_list)
                        next_idx = self.state_to_idx[new_board]
                        self.P[s_idx, a, next_idx] += prob_each
                    
                    self.R[s_idx, a] = 0.0
    
    def to_mdp(self, gamma: float = 0.99) -> MDP:
        """转换为标准 MDP 接口（成本 = -奖励）"""
        return MDP(
            n_states=self.n_states,
            n_actions=self.n_actions,
            gamma=gamma,
            P=self.P,
            C=-self.R  # 成本 = 负奖励
        )
    
    def render(self, board_idx: int):
        """打印棋盘"""
        board = self.all_states[board_idx]
        symbols = {self.EMPTY: '.', self.AI: 'X', self.OPPONENT: 'O'}
        for i in range(self.k):
            row = [symbols[board[i * self.k + j]] for j in range(self.k)]
            print(' '.join(row))
        print()


def tictactoe_mdp(gamma: float = 0.99) -> MDP:
    """
    创建 TicTacToe MDP
    
    参数:
        gamma: 折扣因子
        
    返回:
        MDP 对象，可直接用于所有算法
    """
    env = TicTacToeEnv()
    return env.to_mdp(gamma=gamma)
    

def evaluate_policy_vs_random(policy: np.ndarray, n_games: int = 500, verbose: bool = True) -> Tuple[float, float, float]:
    """
    评估策略与随机对手对战的胜率
    
    参数:
        policy: 策略数组，policy[state_idx] = action
        n_games: 对局数
        verbose: 是否打印结果
        
    返回:
        (win_rate, lose_rate, draw_rate)
    """
    env = TicTacToeEnv()
    
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
            
            new_board_list = list(board)
            new_board_list[action] = current_player
            board = tuple(new_board_list)
        
        if env._check_win(board, env.AI):
            wins += 1
        elif env._check_win(board, env.OPPONENT):
            losses += 1
        else:
            draws += 1
    
    win_rate = wins / n_games * 100
    lose_rate = losses / n_games * 100
    draw_rate = draws / n_games * 100
    
    if verbose:
        print(f"  胜率: {win_rate:.1f}% ({wins}/{n_games})")
        print(f"  负率: {lose_rate:.1f}% ({losses}/{n_games})")
        print(f"  平率: {draw_rate:.1f}% ({draws}/{n_games})")
    
    return win_rate, lose_rate, draw_rate


# ============================================================
# 3. 主实验脚本
# ============================================================
def analyze_nash_and_strategy(policy, env, algorithm_name):
    """
    针对 TicTacToe 分析：
    1. 对于随机对手，AI 的最优策略（起点走法）
    2. 理论 Nash 均衡：双方最优 -> 平局
    """
    print(f"\n{'='*60}")
    print(f"{algorithm_name} - 策略与Nash均衡分析")
    print(f"{'='*60}")

    # --- 1. 针对随机对手的最优策略（起点走法）---
    empty_board = tuple([env.EMPTY] * env.board_size)
    start_idx = env.state_to_idx[empty_board]
    best_start_action = int(policy[start_idx])
    
    print(f"1. 针对随机对手的最优策略 (算法找到):")
    print(f"   空棋盘时 AI 的第一步下在位置: {best_start_action}")
    
    # 统计策略稳定性（胜率已在前面的 evaluate_policy_vs_random 中得到）
    # 这里只做定性理论分析

    # --- 2. 理论 Nash 均衡（双方最优）---
    print(f"\n2. 理论上的 Nash 均衡 (双方都最优):")
    print(f"   在标准 3x3 井字棋中，若双方都采用最优策略（极小化极大），")
    print(f"   最终结果一定是平局。")
    print(f"   这就是该博弈的 Nash 均衡值: 0（对先手而言，无法保证赢，只能保证不输）")
    print(f"   算法找到的针对随机对手的策略是最大化胜率，但并非极小化极大策略，")
    print(f"   因此不是严格意义上的博弈论 Nash 均衡。")
    
    # --- 3. 如果要看极小化极大下的 Nash 策略，可以给出一个简单提示 ---
    print(f"\n3. 极小化极大下的 Nash 均衡策略特点（理论）:")
    print(f"   先手最优第一步: 角或中心")
    print(f"   后手最优第一步: 中心或角")
    print(f"   完美对局 → 平局")
    
    return best_start_action



def run_tictactoe_experiment():
    print("="*70)
    print("TicTacToe MDP 实验")
    print("所有算法 vs 随机对手 + Nash均衡分析")
    print("="*70)
    
    mdp = tictactoe_mdp(gamma=0.99)
    mdp.nS = mdp.n_states
    mdp.nA = mdp.n_actions
    mdp.shape = (1, mdp.n_states)
    
    # 原始环境（用于策略分析）
    raw_env = TicTacToeEnv()
    
    algorithms = {
        'Value Iteration': ValueIterationAgent,
        'RandomVI': RandomVIAgent,
        'Influence Tree': InfluenceTreeAgent,
        'CyclicVI': CyclicVIAgent,
        'RPCyclicVI': RPCyclicVIAgent,
        'Policy Iteration': PolicyIterationAgent,
        'Q-Learning': QLearningAgent,
    }
    
    results = {}
    
    for name, agent_class in algorithms.items():
        print(f"\n{'='*50}")
        print(f"运行 {name}...")
        print(f"{'='*50}")
        
        start_time = time.time()
        agent = agent_class(mdp)
        
        try:
            if name == 'Q-Learning':
                policy, values = agent.optimize(episodes=2000, max_steps_per_episode=20)
            else:
                policy, values = agent.optimize(theta=1e-4)
            
            runtime = time.time() - start_time
            iterations = agent.round_num
            
            print(f"  收敛时间: {runtime:.2f} 秒")
            print(f"  迭代次数: {iterations}")
            
            # 评估胜率（已有函数）
            print(f"\n  策略评估 (500 局 vs 随机对手):")
            win_rate, lose_rate, draw_rate = evaluate_policy_vs_random(policy, n_games=500)
            
            # 新增：分析 Nash 均衡与策略
            best_start = analyze_nash_and_strategy(policy, raw_env, name)
            
            results[name] = {
                'iterations': iterations,
                'runtime': runtime,
                'win_rate': win_rate,
                'lose_rate': lose_rate,
                'draw_rate': draw_rate,
                'best_start': best_start
            }
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results[name] = None
    
    # 打印汇总表格（可以增加一列 Best Start）
    print("\n" + "="*70)
    print("实验结果汇总（含Nash分析）")
    print("="*70)
    print(f"{'Algorithm':<20} {'Iter':<8} {'Time(s)':<8} {'Win%':<8} {'Best Start':<10}")
    print("-"*70)
    for name, res in results.items():
        if res:
            print(f"{name:<20} {res['iterations']:<8} {res['runtime']:<8.2f} {res['win_rate']:<8.1f} {res['best_start']:<10}")
        else:
            print(f"{name:<20} {'FAILED':<8}")
    
    return results


if __name__ == "__main__":
    results = run_tictactoe_experiment()