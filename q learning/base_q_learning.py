import sys
import random
import time as _time
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ALPHA            = 0.1
GAMMA            = 0.9
EPSILON_KEY      = "e1"
REWARD_KEY       = "r4"
DEAD_END_PENALTY = -50.0
EXPLORE_POWER    = 10

def parse_input():
    data = sys.stdin.read().replace('\r', '').split()
    if not data:
        return None, None, None, None, None
    idx  = 0
    N    = int(data[idx]); idx += 1
    e    = [0]     * (N + 1)
    l    = [10**9] * (N + 1)
    d    = [0]     * (N + 1)
    for i in range(1, N + 1):
        e[i] = int(data[idx]); idx += 1
        l[i] = int(data[idx]); idx += 1
        d[i] = int(data[idx]); idx += 1
    t = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            t[i][j] = int(data[idx]); idx += 1
    return N, e, l, d, t

class BaseQLearning:
    def __init__(self, N, e, l, d, t,
                 alpha=ALPHA, gamma=GAMMA,
                 epsilon_key=EPSILON_KEY, reward_key=REWARD_KEY,
                 episodes=5000):
        self.N           = N
        self.e           = e
        self.l           = l
        self.d           = d
        self.t           = t
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon_key = epsilon_key
        self.reward_key  = reward_key
        self.episodes    = episodes
        self.episode     = 0
        self.starting_node = 0
    
    #Chiến lược epsilon decay
    @property
    def epsilon(self) -> float:
        ep, eps = self.episode, self.episodes
        if self.epsilon_key == "e1": #tuyến tính
            return 1 - ep / eps
        elif self.epsilon_key == "e2": #Cấp số nhân
            return 0.999 ** ep
        elif self.epsilon_key == "e3": #đa thức bậc 6
            return -((ep / eps) ** 6) + 1
        elif self.epsilon_key == "e4": #Giảm 0.1 mỗi 10% số episode
            return 1 - (0.1 * (ep // (eps // 10)))
        raise ValueError(f"Unknown epsilon key: {self.epsilon_key}")

    #Tính phần thưởng dựa trên khoảng cách và thời gian
    def reward(self, i: int, j: int, cur_t: float) -> float:
        dist = self.t[i][j]
        if self.reward_key == "r1": #Nghich đảo khoảng cách
            return (1.0 / dist) if dist > 0 else 10.0
        elif self.reward_key == "r2": #Phạt theo khoảng cách âm
            return -dist
        elif self.reward_key == "r3": #Phạt theo bình phương khoảng cách âm
            return -(dist ** 2)
        elif self.reward_key == "r4": #Thưởng nếu đến trước hạn, phạt nếu đến muộn
            slack = self.l[j] - (cur_t + dist)
            return slack / max(1, self.l[j])
        raise ValueError(f"Unknown reward key: {self.reward_key}")

    """Chọn địa điểm tiếp theo để khán phá dựa trên độ ưu tiên
       - Sắp xếp các địa điểm theo thứ tự tăng dần của l[j] (địa điểm có hạn chót sớm hơn được ưu tiên hơn)
       - Tạo trọng số cho mỗi địa điểm dựa trên thứ tự ưu tiên (địa điểm ưu tiên cao hơn có trọng số lớn hơn)
       - Chọn ngẫu nhiên một địa điểm dựa trên phân phối trọng số này (địa điểm ưu tiên cao hơn có khả năng được chọn cao hơn)
    """
    @staticmethod
    def explore(environment: set, l: list) -> int:
        feas_sorted = sorted(environment, key=lambda j: l[j])
        n = len(feas_sorted)
        weights = [(n - i) ** EXPLORE_POWER for i in range(n)]
        total   = sum(weights)
        r       = random.random() * total
        cum     = 0.0
        for city, w in zip(feas_sorted, weights):
            cum += w
            if cum >= r:
                return city
        return feas_sorted[0]

    def exploit(self, s: int, environment: set) -> int:
        raise NotImplementedError

    def next_action(self, state: int, environment: set,
                    allow_exploration: bool = True) -> int:
        if allow_exploration and random.random() < self.epsilon:
            return self.explore(environment, self.l)
        return self.exploit(state, environment)

    def update_Q_table(self, state, action, reward_val, a_t1):
        raise NotImplementedError

    def _feasible_env(self, node: int, cur_t: float, unvisited: set) -> set:
        return {j for j in unvisited if cur_t + self.t[node][j] <= self.l[j]}

    def _service_start(self, cur_t: float, fr: int, to: int) -> float:
        return max(cur_t + self.t[fr][to], self.e[to])


    """"- Tính action a, reward r, và next action khi đã ở a: a_t1 để cập nhật Q-table
        - Dùng khi train""" 
    def q_learning(self):
        N = self.N
        starting_node = self.starting_node
        best_cost  = float('inf')
        best_route = None

        for self.episode in range(self.episodes):

            cur_t    = 0.0
            route    = [starting_node]
            ep_cost  = 0.0
            dead_end = False

            while len(route) < N + 1:
                state       = route[-1] #Vị trí hiện tại
                unvisited   = set(range(1, N + 1)) - set(route)
                environment = self._feasible_env(state, cur_t, unvisited)

                if not environment:
                    if len(route) >= 2:
                        prev, curr = route[-2], route[-1]
                        self.update_Q_table(prev, curr, DEAD_END_PENALTY, starting_node)
                    dead_end = True
                    break

                action = self.next_action(state, environment) #Có thể là explore hoặc exploit tùy theo epsilon
                r      = self.reward(state, action, cur_t)

                cur_t = self._service_start(cur_t, state, action) + self.d[action] 

                new_unvisited   = unvisited - {action}
                new_environment = self._feasible_env(action, cur_t, new_unvisited)
                if new_environment:
                    a_t1 = self.next_action(action, new_environment)
                else:
                    a_t1 = starting_node

                self.update_Q_table(state, action, r, a_t1)

                ep_cost += self.t[state][action]
                route.append(action)

            if dead_end:
                continue

            last   = route[-1]
            action = starting_node
            r_last = self.reward(last, action, cur_t)

            depot_env = self._feasible_env(starting_node, 0.0, set(range(1, N + 1)))
            a_t1 = self.next_action(starting_node, depot_env) if depot_env else starting_node

            self.update_Q_table(last, action, r_last, a_t1)
            ep_cost += self.t[last][starting_node]

            if ep_cost < best_cost:
                best_cost  = ep_cost
                best_route = route[1:]

        return best_cost, best_route

    #Sau khi train xong, dùng Q-table để chọn hành động tốt nhất (không còn explore nữa) để tạo route hoàn chỉnh"""
    def algorithm(self):
        N             = self.N
        starting_node = self.starting_node

        train_cost, train_route = self.q_learning()

        route    = [starting_node]
        cur_t    = 0.0
        cost     = 0.0
        feasible = True

        while len(route) < N + 1:
            state       = route[-1]
            unvisited   = set(range(1, N + 1)) - set(route)
            environment = self._feasible_env(state, cur_t, unvisited)

            if not environment:
                feasible = False
                break

            action = self.next_action(state, environment, allow_exploration=False)
            cost  += self.t[state][action]
            cur_t  = self._service_start(cur_t, state, action) + self.d[action]
            route.append(action)

        if feasible:
            cost += self.t[route[-1]][starting_node]
            greedy_cost  = cost
            greedy_route = route[1:]
        else:
            greedy_cost  = float('inf')
            greedy_route = None

        if greedy_route is not None and greedy_cost < train_cost:
            return greedy_cost, greedy_route
        return train_cost, train_route