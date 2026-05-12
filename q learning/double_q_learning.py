import time as _time
import numpy as np
import random
from base_q_learning import BaseQLearning, parse_input, ALPHA, GAMMA, EPSILON_KEY, REWARD_KEY
EPISODES = 30000

class DoubleQLearning(BaseQLearning):
    algorithm_name = "Double Q-Learning"
    abbreviation   = "dq"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q_a = np.zeros((self.N + 1, self.N + 1))
        self.Q_b = np.zeros((self.N + 1, self.N + 1))

    #Bảng Q có hàng là state, cột là action, giá trị mỗi ô là Q-value của cặp (state, action) đó.
    def update_Q_table(self, state: int, action: int, reward_val: float, a_t1: int) -> None:
        #Xác suất 50% để cập nhật Q_a hoặc Q_b

        #Cập nhật Q_a dựa trên Q_b
        if random.random() < 0.5: 
            Q_b_t1 = self.Q_b[action, a_t1]
            tdt_a  = reward_val + self.gamma * Q_b_t1 - self.Q_a[state, action]
            self.Q_a[state, action] += self.alpha * tdt_a

        #Cập nhật Q_b dựa trên Q_a
        else: 
            Q_a_t1 = self.Q_a[action, a_t1]
            tdt_b  = reward_val + self.gamma * Q_a_t1 - self.Q_b[state, action]
            self.Q_b[state, action] += self.alpha * tdt_b

    def exploit(self, s: int, environment: set) -> int:
        max_reward = -np.inf
        best_a     = None
        #Duyêt qua tất cả các hành động có thể và chọn hành động có Q-value cao nhất (trung bình của Q_a và Q_b)
        for a in environment:
            r = np.average([self.Q_a[s, a], self.Q_b[s, a]])
            if r > max_reward:
                max_reward = r
                best_a     = a
        return best_a

def main():
    N, e, l, d, t = parse_input()
    if N is None:
        return

    episodes = EPISODES

    solver = DoubleQLearning(
        N=N, e=e, l=l, d=d, t=t,
        alpha=ALPHA, gamma=GAMMA,
        epsilon_key=EPSILON_KEY,
        reward_key=REWARD_KEY,
        episodes=episodes,
    )

    cost, route = solver.algorithm()

    if route is None:
        print("-1.0")
    else:
        print(float(cost))

if __name__ == "__main__":
    main()