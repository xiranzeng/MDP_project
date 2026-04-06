import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class Env(object):
    def step(self, a):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class DiscreteEnv(Env):
    def __init__(self, nS, nA, P, isd):
        self.nS = nS  # num of state
        self.nA = nA  # num of action
        self.P = P  # transition probs, P[s][a] == [(prob, s', r, done), ...]
        self.isd = isd  # isd: initial state distribution
        self.reset()

    def step(self, a):
        transitions = self.P[self.s][a]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        p, s, r, d = transitions[i]
        self.s = s
        return (s, r, d, {"prob": p})

    def reset(self):
        self.s = np.random.choice(self.nS, p=self.isd)
        return self.s


class GridWorld(DiscreteEnv):
    def __init__(self, shape=[3, 3], target=[1, 2], wind_prob=.0):
        self.shape = shape
        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        target = target[0] + target[1] * MAX_X

        P = {}
        grid = np.arange(nS).reshape(shape)

        iterator = np.nditer(grid, flags=['multi_index'])
        while not iterator.finished:
            s = iterator.iterindex
            y, x = iterator.multi_index

            P[s] = {a: [] for a in range(nA)}

            is_done = (lambda s: s == target)
            reward = 0.0 if is_done(s) else - 1.0

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_up_wind = ns_up if y <= 1 else ns_up - MAX_X

                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_right_wind = ns_right if y == 0 else ns_right - MAX_X

                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_down_wind = s

                ns_left = s if x == 0 else s - 1
                ns_left_wind = ns_left if y == 0 else ns_left - MAX_X

                P[s][UP] = [(1.0 - wind_prob, ns_up, reward, is_done(ns_up)),
                            (wind_prob, ns_up_wind, reward, is_done(ns_up_wind))]
                P[s][RIGHT] = [(1.0 - wind_prob, ns_right, reward, is_done(ns_right)),
                               (wind_prob, ns_right_wind, reward, is_done(ns_right_wind))]
                P[s][DOWN] = [(1.0 - wind_prob, ns_down, reward, is_done(ns_down)),
                              (wind_prob, ns_down_wind, reward, is_done(ns_down_wind))]
                P[s][LEFT] = [(1.0 - wind_prob, ns_left, reward, is_done(ns_left)),
                              (wind_prob, ns_left_wind, reward, is_done(ns_left_wind))]

            iterator.iternext()

        isd = np.ones(nS) / (nS - 1)
        isd[target] = 0

        super(GridWorld, self).__init__(nS, nA, P, isd)

    def get_action_name(self, a):
        action2name = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}
        return action2name[a]
    
    
class GamblersMDP:
    """赌博问题：赌徒希望通过赌博达到目标金额"""
    def __init__(self, goal=100, p_head=0.4):
        self.goal = goal
        self.p_head = p_head
        self.nS = goal + 1  # 状态：0 到 100（金额）
        self.nA = min(100, 50)  # 动作：赌注金额（简化版）
        self.shape = (1, self.nS)
        
        self.P = {}
        for s in range(self.nS):
            self.P[s] = {a: [] for a in range(self.nA)}
            
            for a in range(min(s, self.nA)):  # 赌注不能超过当前金额
                # 赢：概率p_head
                win_state = min(s + a, self.goal)
                win_reward = 1 if win_state == self.goal else 0
                self.P[s][a].append((self.p_head, win_state, win_reward, win_state == self.goal))
                
                # 输：概率1-p_head
                lose_state = s - a
                lose_reward = 0
                self.P[s][a].append((1 - self.p_head, lose_state, lose_reward, False))
            
            # 填充未定义的动作（不合法赌注）
            for a in range(min(s, self.nA), self.nA):
                self.P[s][a] = [(1.0, s, 0, False)]
        
        # 终止状态
        self.P[self.goal] = {a: [(1.0, self.goal, 0, True)] for a in range(self.nA)}
    
    def get_action_name(self, a):
        return f"Bet_{a}"
    

class FrozenLakeMDP:
    """简化的冰湖问题"""
    def __init__(self, size=4):
        self.size = size
        self.nS = size * size
        self.nA = 4  # 上下左右
        self.shape = (size, size)
        
        # 定义冰洞位置（简化版）
        self.holes = [5, 7, 11, 12]
        self.goal = 15
        
        self.P = {}
        for s in range(self.nS):
            self.P[s] = {a: [] for a in range(self.nA)}
            
            if s == self.goal or s in self.holes:
                # 终止状态
                for a in range(self.nA):
                    self.P[s][a] = [(1.0, s, 0, True)]
            else:
                # 计算每个动作的下一状态
                row, col = divmod(s, size)
                
                for a in range(self.nA):
                    # 定义移动
                    if a == 0:  # UP
                        next_s = s if row == 0 else s - size
                    elif a == 1:  # DOWN
                        next_s = s if row == size-1 else s + size
                    elif a == 2:  # LEFT
                        next_s = s if col == 0 else s - 1
                    else:  # RIGHT
                        next_s = s if col == size-1 else s + 1
                    
                    # 确定奖励和终止
                    if next_s == self.goal:
                        reward = 1.0
                        done = True
                    elif next_s in self.holes:
                        reward = -1.0
                        done = True
                    else:
                        reward = -0.01  # 小惩罚鼓励快速到达
                        done = False
                    
                    self.P[s][a] = [(0.8, next_s, reward, done)]  # 确定性移动
                    # 可以添加滑动概率使问题更复杂
    
    def get_action_name(self, a):
        return ['U', 'D', 'L', 'R'][a]

