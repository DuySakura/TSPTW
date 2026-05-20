"""
Deep Q-Learning for TSPTW (Traveling Salesman Problem with Time Windows)
  BaseQLearning          (base_q_learning.py — imported unchanged)
  └── DeepQLearning      (this file)
       ├── DQNetwork     — 2-hidden-layer ReLU MLP, numpy + Adam (no torch needed)
       ├── ReplayBuffer  — circular experience buffer for mini-batch training
       ├── exploit()     — policy_net forward pass, masked to feasible cities
       └── update_Q_table() — add (s,a,r,s') to buffer; train policy_net on batch;
                              periodically sync target_net ← policy_net

DQN-specific components (replacing Q_a/Q_b tables):
  policy_net   — trained online, selects actions during exploitation
  target_net   — frozen copy of policy_net, updated every TARGET_UPDATE steps
                 provides stable TD targets (reduces oscillation)
  ReplayBuffer — stores past transitions; random mini-batch breaks correlation

TSPTW adaptations (same as base_q_learning.py):
  • Environment filtered to time-window-feasible cities at every step
  • Reward r4: normalised remaining slack (guides toward tight-deadline cities)
  • Rank-weighted stochastic exploration (EXPLORE_POWER=10)
  • Dead-end penalty on the transition that caused the dead-end
  • Wall-clock DEADLINE respected inside q_learning()

State encoding — fixed-size float32 vector of length 2 + 3·N:
  [0]   cur_node / N                      (normalised current position)
  [1]   cur_time / max_deadline           (normalised elapsed time)
  [2+3·(j-1)]   unvisited_flag[j]         ∈ {0,1}
  [2+3·(j-1)+1] arrival_time[j] / max_l   normalised arrival if we go to j next
  [2+3·(j-1)+2] slack[j] / max_l          normalised slack remaining at j

Action: integer in {1…N} (customer index to visit next)
Output: Q-value for each of the N+1 nodes (0=depot masked out during tour)
"""

import random
import time as _time
from collections import deque
import os

import numpy as np

from base_q_learning import (
    BaseQLearning, parse_input,
    ALPHA, GAMMA, EPSILON_KEY, REWARD_KEY, DEAD_END_PENALTY,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HIDDEN_SIZE        = 128   
LEARNING_RATE      = 1e-3  
BATCH_SIZE         = 64     
BUFFER_SIZE        = 8_000 
TARGET_UPDATE_FREQ = 200    # copy policy_net → target_net every N train steps
MIN_BUFFER_FILL    = 256    # start training only after this many transitions
EPISODES           = 5_000  # default episode budget (overridden in main())


#  2-hidden-layer ReLU MLP with Adam optimiser
class DQNetwork:
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 lr: float = LEARNING_RATE):
        self.out_dim = out_dim
        self.lr      = lr

        # He (Kaiming) weight initialisation
        s1 = np.sqrt(2.0 / in_dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = (np.random.randn(in_dim, hidden) * s1).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (np.random.randn(hidden, hidden) * s2).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = (np.random.randn(hidden, out_dim) * s2).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

        # Adam first/second moment accumulators
        self._adam_t = 0
        self._beta1  = 0.9
        self._beta2  = 0.999
        self._eps    = 1e-8
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            p = getattr(self, name)
            setattr(self, "_m" + name, np.zeros_like(p))
            setattr(self, "_v" + name, np.zeros_like(p))

    #  forward (single sample)
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (in_dim,) → q: (out_dim,)"""
        h1 = np.maximum(0.0, x  @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    #  forward (batch)
    def _forward_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (B, in_dim) → Q: (B, out_dim).  Caches activations for backward."""
        self._cache_X  = X
        self._cache_h1 = np.maximum(0.0, X  @ self.W1 + self.b1)
        self._cache_h2 = np.maximum(0.0, self._cache_h1 @ self.W2 + self.b2)
        return self._cache_h2 @ self.W3 + self.b3

    #  Adam helper
    def _adam_step(self, param, grad, m, v):
        m[:] = self._beta1 * m + (1.0 - self._beta1) * grad
        v[:] = self._beta2 * v + (1.0 - self._beta2) * grad * grad
        m_hat = m / (1.0 - self._beta1 ** self._adam_t)
        v_hat = v / (1.0 - self._beta2 ** self._adam_t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps)

    #  training step 
    def train_step(self, X: np.ndarray,
                   targets: np.ndarray,
                   action_idx: np.ndarray) -> float:
        """
        One mini-batch gradient descent step.

        X          : (B, in_dim) — state batch
        targets    : (B,)        — TD target  r + γ · max_a Q_target(s', a)
        action_idx : (B,)        — index of the action taken (int)

        Loss: MSE only on the Q-value of the taken action (standard DQN).
        Returns mean squared error (for logging).
        """
        B  = X.shape[0]
        Q  = self._forward_batch(X)                           # (B, out_dim)

        # Gradient of MSE loss w.r.t. Q-output (zero for all but taken action)
        errors = Q[np.arange(B), action_idx] - targets       # (B,)
        dQ     = np.zeros_like(Q)
        dQ[np.arange(B), action_idx] = 2.0 * errors / B

        # Backprop through layer 3
        dh2 = (dQ @ self.W3.T) * (self._cache_h2 > 0.0)
        # Backprop through layer 2
        dh1 = (dh2 @ self.W2.T) * (self._cache_h1 > 0.0)

        self._adam_t += 1
        self._adam_step(self.W3, self._cache_h2.T @ dQ / B, self._mW3, self._vW3)
        self._adam_step(self.b3, dQ.mean(axis=0),            self._mb3, self._vb3)
        self._adam_step(self.W2, self._cache_h1.T @ dh2 / B, self._mW2, self._vW2)
        self._adam_step(self.b2, dh2.mean(axis=0),            self._mb2, self._vb2)
        self._adam_step(self.W1, self._cache_X.T  @ dh1 / B, self._mW1, self._vW1)
        self._adam_step(self.b1, dh1.mean(axis=0),            self._mb1, self._vb1)

        return float(np.mean(errors ** 2))

    #  target network sync
    def copy_weights_from(self, other: "DQNetwork") -> None:
        """Hard-copy weights from another network (target net ← policy net)."""
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            getattr(self, name)[:] = getattr(other, name)


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf  = deque(maxlen=capacity) #Tạo double-end queue lưu ký ức với kích thước tối đa là capacity

    #push khi thực hiện một bước đi, nếu buffer đầy thì loại bỏ phần tử cũ nhất
    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, float(reward), next_state, bool(done)))

    #Bốc n phần tử ngẫu nhiên từ buffer
    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# DeepQLearning  
class DeepQLearning(BaseQLearning):
    """
    Deep Q-Learning for TSPTW.

    Replaces the tabular Q_a / Q_b arrays from DoubleQLearning with:
      policy_net  — DQNetwork trained online via mini-batch gradient descent
      target_net  — frozen copy updated every TARGET_UPDATE_FREQ train steps
      replay      — ReplayBuffer for experience replay

    All logic inherited from BaseQLearning (epsilon schedule, reward function,
    explore(), next_action(), q_learning(), algorithm()) is reused unchanged.
    Only exploit() and update_Q_table() are overridden.
    """

    algorithm_name = "Deep Q-Learning"
    abbreviation   = "dq_deep"

    def __init__(self, *args,
                 hidden_size: int    = HIDDEN_SIZE,
                 lr: float           = LEARNING_RATE,
                 batch_size: int     = BATCH_SIZE,
                 buffer_size: int    = BUFFER_SIZE,
                 target_update: int  = TARGET_UPDATE_FREQ,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size    = batch_size
        self.target_update = target_update

        # Chuẩn hoá bằng cách chia thời gian đến hạn l[i] cho max_l để giữ ở khoảng [0,1]
        self._max_l = max(self.l[1:self.N + 1]) if self.N > 0 else 1

        """ Tính số neuron đầu vào: 
         - người giao hàng 2 đặc trưng: vị trí hiện tại (cur_node / N) và thời gian hiện tại (cur_t / _max_l)
         - Mỗi khách hàng có 3 đặc trưng: trạng thái viếng thăm(0/1), thòi gian dự kiến đến, thời gian dư giả (slack)
        """
        self._state_dim = 2 + 3 * self.N

        # Build policy net and target net (same architecture)
        self.policy_net = DQNetwork(self._state_dim, hidden_size, self.N + 1, lr)
        self.target_net = DQNetwork(self._state_dim, hidden_size, self.N + 1, lr)
        self.target_net.copy_weights_from(self.policy_net)

        if os.path.exists('dqn_weights_N10.npz'):
            self.policy_net.load('dqn_weights_N10.npz')
            self.target_net.copy_weights_from(self.policy_net)

        self.replay       = ReplayBuffer(buffer_size)
        self._train_steps = 0   
        # Cache the last (state_vec, cur_t, cur_node, unvisited) for transition storage
        self._last_state_vec  = None
        self._last_action     = None

    # state encoding 
    def _encode(self, cur_node: int, cur_t: float, unvisited: set) -> np.ndarray:
        """
        Build the fixed-size float32 feature vector for the current step.

        Features:
          [0]          cur_node / N               (normalised position)
          [1]          cur_t / max_deadline        (normalised time)
          For j=1..N:
            [2+3(j-1)] 1 if j ∈ unvisited else 0
            [2+3(j-1)+1] (cur_t + t[cur_node][j]) / max_l  (normalised arrival)
            [2+3(j-1)+2] max(0, l[j] - cur_t - t[cur_node][j]) / max_l (slack)
        """
        N     = self.N
        max_l = self._max_l
        feat  = np.zeros(self._state_dim, dtype=np.float32)
        feat[0] = cur_node / max(1, N) #Vị trí hiện tại được chuẩn hóa bằng cách chia cho N để giữ ở khoảng [0,1]
        feat[1] = cur_t    / max(1, max_l) #Thời gian hiện tại được chuẩn hóa bằng cách chia cho max_l (thời gian đến hạn lớn nhất) để giữ ở khoảng [0,1]
        t_row   = self.t[cur_node]
        for j in range(1, N + 1): #Duyệt qua từng khách hàng
            base         = 2 + 3 * (j - 1)
            feat[base]   = 1.0 if j in unvisited else 0.0
            arrival      = cur_t + t_row[j]
            feat[base+1] = arrival           / max(1, max_l)
            feat[base+2] = max(0.0, self.l[j] - arrival) / max(1, max_l)
        return feat

    # exploit (overrides BaseQLearning) 
    def exploit(self, s: int, environment: set) -> int:
        """
        Choose the action with the highest Q-value from policy_net.
        Infeasible cities are masked to −∞ before argmax.

        s : current node (used only to recover cur_t; stored via _last_state_vec)
        environment: set of feasible (time-window-respecting) cities
        """
        # Recover current state vector (set by q_learning loop before calling next_action)
        state_vec = self._last_state_vec
        if state_vec is None:
            # Fallback: pick randomly (should not happen in normal flow)
            return random.choice(list(environment))

        #Đưa trạng thái vào mạng policy_net để tính Q-values cho tất cả các hành động (địa điểm tiếp theo)
        q_values = self.policy_net.forward(state_vec)       # (N+1,)

        # Loại bỏ các đỉnh không khả thi bằng cách gán Q-value của chúng thành -inf, sau đó chọn hành động có Q-value cao nhất
        masked = np.full(self.N + 1, -np.inf, dtype=np.float32)
        for a in environment:
            masked[a] = q_values[a]

        return int(np.argmax(masked))

    # update_Q_table (overrides BaseQLearning) 
    def update_Q_table(self, state: int, action: int,
                       reward_val: float, a_t1: int) -> None:
        """
        Store transition in replay buffer and perform one DQN training step.

        Replaces the tabular Double-Q update from double_q_learning.py with:
          1. Push (s, a, r, s') to the replay buffer.
          2. If buffer has enough samples: sample a mini-batch and train
             policy_net using the DQN target:
               y = r + γ · max_{a'} Q_target(s', a')   (non-terminal)
               y = r                                    (terminal)
          3. Every TARGET_UPDATE_FREQ steps: hard-copy policy_net → target_net.

        `state`    and `a_t1` are node indices, but the transition is stored
        as raw feature vectors (self._last_state_vec already set by caller).
        """
        # Lấy vector trạng thái hiện tại
        s_vec = self._last_state_vec
        if s_vec is None:
            return

        # Lấy vector trạng thái tiếp theo 
        next_s_vec = self._next_state_vec
        done = (next_s_vec is None)       # Nếu done thì chuyến đi đã kết thúc, không có trạng thái tiếp theo, 
        #nên vector trạng thái tiếp theo sẽ toàn số 0
        if done:
            next_s_vec = np.zeros(self._state_dim, dtype=np.float32)
        
        #đẩy vào trong replay bufer
        self.replay.push(s_vec, action, reward_val, next_s_vec, done)

        # Train only when buffer is large enough
        if len(self.replay) < MIN_BUFFER_FILL:
            return

        # Sample mini-batch (states, actions, rewards, next_states, dones)
        S, A, R, S2, D = self.replay.sample(min(self.batch_size, len(self.replay)))

        # DQN target:  y = R + gamma . Q_{next} . (1 - D)
        #y là giá trị mục tiêu mà policy_net cố gắng dự đoán
        # R là phần thưởng thực tế nhận được sau khi thực hiện hành động A từ trạng thái S
        # Q_{next} là giá trị Q tối đa cho trạng thái tiếp theo S2
        # D là biến chỉ thị xem trạng thái tiếp theo có phải là trạng thái kết thúc không
        # gamma là hệ số chiết khấu, điều chỉnh tầm quan trọng của phần thưởng tương lai so với phần thưởng hiện tại
        Q_next   = np.max(self.target_net._forward_batch(S2), axis=1)   
        targets  = R + self.gamma * Q_next * (1.0 - D)                 

        self.policy_net.train_step(S, targets, A)
        self._train_steps += 1

        # Periodically sync target network (hard update)
        if self._train_steps % self.target_update == 0:
            self.target_net.copy_weights_from(self.policy_net)

    # override q_learning to inject state caching and deadline guard
    def q_learning(self):
        """
        Training loop — reuses BaseQLearning.q_learning() logic verbatim, but
        wraps each step with:
          (a) self._last_state_vec ← encoded state before selecting action
          (b) self._next_state_vec ← encoded state after transition (for replay)
          (c) Wall-clock DEADLINE check at start of each episode.
        """
        N             = self.N
        starting_node = self.starting_node
        best_cost     = float('inf')
        best_route    = None

        for self.episode in range(self.episodes):
            # wall-clock deadline (TSPTW grader constraint) 
            if self.deadline and _time.time() > self.deadline:
                break

            cur_t    = 0.0
            route    = [starting_node]
            ep_cost  = 0.0
            dead_end = False
            unvisited = set(range(1, N + 1))

            while len(route) < N + 1:
                state        = route[-1]
                environment  = self._feasible_env(state, cur_t, unvisited)

                if not environment:
                    if len(route) >= 2:
                        prev, curr = route[-2], route[-1]
                        # Cache state vectors for penalty update
                        self._last_state_vec = self._encode(prev, 0.0, unvisited | {curr})
                        self._next_state_vec = None
                        self.update_Q_table(prev, curr,
                                            DEAD_END_PENALTY, starting_node)
                    dead_end = True
                    break

                # Encode current state BEFORE choosing action
                self._last_state_vec = self._encode(state, cur_t, unvisited)

                action = self.next_action(state, environment)

                r = self.reward(state, action, cur_t)

                cur_t = self._service_start(cur_t, state, action) + self.d[action]
                unvisited = unvisited - {action}

                # Encode next state (after transition) for replay storage
                if unvisited:
                    self._next_state_vec = self._encode(action, cur_t, unvisited)
                else:
                    self._next_state_vec = None  # terminal

                # Look-ahead action for DQN target (a_{t+1} from next state)
                new_environment = self._feasible_env(action, cur_t, unvisited)
                if new_environment:
                    # Temporarily set _last_state_vec to next state for look-ahead
                    _saved = self._last_state_vec
                    self._last_state_vec = self._next_state_vec
                    a_t1 = self.next_action(action, new_environment)
                    self._last_state_vec = _saved
                else:
                    a_t1 = starting_node

                self.update_Q_table(state, action, r, a_t1)

                ep_cost += self.t[state][action]
                route.append(action)

            if dead_end:
                continue

            # Close tour: last city → depot
            last   = route[-1]
            action = starting_node
            r_last = self.reward(last, action, cur_t)

            self._last_state_vec = self._encode(last, cur_t, set())
            self._next_state_vec = None   # terminal

            depot_env = self._feasible_env(starting_node, 0.0, set(range(1, N + 1)))
            a_t1 = (self.next_action(starting_node, depot_env)
                    if depot_env else starting_node)

            self.update_Q_table(last, action, r_last, a_t1)
            ep_cost += self.t[last][starting_node]

            if ep_cost < best_cost:
                best_cost  = ep_cost
                best_route = route[1:]

        return best_cost, best_route

def main():
    N, e, l, d, t = parse_input()

    solver = DeepQLearning(
        N=N, e=e, l=l, d=d, t=t,
        alpha=ALPHA, gamma=GAMMA,
        epsilon_key=EPSILON_KEY,
        reward_key=REWARD_KEY,
        episodes=EPISODES,
    )

    cost, route = solver.algorithm()

    if route is None:
        print(-1.0)
    else:
        print(float(cost))


if __name__ == "__main__":
    main()
