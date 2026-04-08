"""
Double Q-Learning for the Traveling Salesman Problem with Time Windows (TSPTW).

Adapted from AGnias47/tsp-research (Double Q-Learning for TSP) and extended for TSPTW.

Key design choices versus the original repo:
  * EDF (Earliest-Deadline-First) greedy is used to
      (a) find a guaranteed initial feasible solution,
      (b) complete any Q-learning episode that reaches a dead-end mid-tour.
    This ensures Q-values are updated every episode, even when random exploration
    wanders into an infeasible partial route.
  * Multiple greedy strategies seed the initial best solution.
  * 2-opt and or-opt local search are applied post-Q-learning to refine.
  * Output is a single float (total travel time) or -1.0 (no feasible route found).
"""

import sys
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ALPHA = 0.1
GAMMA = 0.9


# ── helpers ────────────────────────────────────────────────────────────────────

def feasible_nbrs(node, cur_t, unvisited, l, t):
    return {j for j in unvisited if cur_t + t[node][j] <= l[j]}

def arrive(cur_t, fr, to, e, t):
    return max(cur_t + t[fr][to], e[to])

def compute_cost(route, e, l, d, t):
    nodes = [0] + route + [0]
    cur_t = 0.0; cost = 0.0
    for i in range(len(nodes) - 1):
        fr, to = nodes[i], nodes[i + 1]
        travel = t[fr][to]; cost += travel
        raw = cur_t + travel
        if to != 0 and raw > l[to]:
            return float('inf')
        cur_t = max(raw, e[to]) + d[to]
    return cost


# ── greedy construction ────────────────────────────────────────────────────────

def greedy_construct(N, e, l, d, t, key_fn):
    cur_t = 0.0; node = 0
    unvisited = set(range(1, N + 1))
    route = []; cost = 0.0
    while unvisited:
        cands = feasible_nbrs(node, cur_t, unvisited, l, t)
        if not cands:
            return float('inf'), None
        action = min(cands, key=lambda j: key_fn(node, cur_t, j))
        cost += t[node][action]; route.append(action)
        cur_t = arrive(cur_t, node, action, e, t) + d[action]
        node = action; unvisited -= {action}
    cost += t[node][0]
    return cost, route


def greedy_complete(node, cur_t, unvisited, e, l, d, t):
    """EDF-greedy fallback used to finish a partial Q-learning episode."""
    cur_node = node; cur_time = cur_t
    remaining = set(unvisited); completion = []; cost = 0.0
    while remaining:
        cands = feasible_nbrs(cur_node, cur_time, remaining, l, t)
        if not cands:
            return float('inf'), None
        action = min(cands, key=lambda j: l[j])
        cost += t[cur_node][action]; completion.append(action)
        cur_time = arrive(cur_time, cur_node, action, e, t) + d[action]
        cur_node = action; remaining -= {action}
    cost += t[cur_node][0]
    return cost, completion


def find_initial_solution(N, e, l, d, t):
    strategies = [
        lambda i, tm, j: l[j],
        lambda i, tm, j: t[i][j],
        lambda i, tm, j: l[j] - tm - t[i][j],
        lambda i, tm, j: arrive(tm, i, j, e, t) + d[j],
    ]
    best_cost = float('inf'); best_route = None
    for fn in strategies:
        cost, route = greedy_construct(N, e, l, d, t, fn)
        if route is not None and cost < best_cost:
            best_cost = cost; best_route = route[:]
    return best_cost, best_route


# ── local search ───────────────────────────────────────────────────────────────

def two_opt(route, e, l, d, t, max_iter=500):
    best = route[:]; best_cost = compute_cost(best, e, l, d, t)
    N = len(route)
    for _ in range(max_iter):
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                new = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                nc = compute_cost(new, e, l, d, t)
                if nc < best_cost:
                    best_cost = nc; best = new; improved = True
        if not improved:
            break
    return best_cost, best


def or_opt(route, e, l, d, t):
    best = route[:]; best_cost = compute_cost(best, e, l, d, t)
    N = len(route); improved = True
    while improved:
        improved = False
        for i in range(N):
            node = best[i]
            candidate = best[:i] + best[i+1:]
            for j in range(N):
                new = candidate[:j] + [node] + candidate[j:]
                nc = compute_cost(new, e, l, d, t)
                if nc < best_cost:
                    best_cost = nc; best = new; improved = True; break
            if improved:
                break
    return best_cost, best


# ── Q-table helpers ────────────────────────────────────────────────────────────

def exploit(s, environment, Q_a, Q_b):
    best_a, best_q = None, -np.inf
    for a in environment:
        q = 0.5 * (Q_a[s, a] + Q_b[s, a])
        if q > best_q:
            best_q = q; best_a = a
    return best_a


def next_action(s, environment, Q_a, Q_b, eps, allow_exploration=True):
    if allow_exploration and random.random() < eps:
        return random.choice(list(environment))
    return exploit(s, environment, Q_a, Q_b)


def update_Q(state, action, reward, a_t1, Q_a, Q_b):
    """
    Symmetric Double Q-Learning update (from AGnias47/tsp-research).
    Each table is updated using the other's value estimate to reduce
    maximisation bias.
    """
    Q_a_t1 = Q_a[action, a_t1]
    Q_b_t1 = Q_b[action, a_t1]
    td_a = reward + GAMMA * Q_b_t1 - Q_a[state, action]
    td_b = reward + GAMMA * Q_a_t1 - Q_b[state, action]
    Q_a[state, action] += ALPHA * td_a
    Q_b[state, action] += ALPHA * td_b


def reward_r1(travel_time):
    return 10.0 if travel_time == 0 else 1.0 / travel_time


# ── Double Q-Learning main loop ────────────────────────────────────────────────

def run_double_q_learning(N, e, l, d, t, episodes, init_cost, init_route):
    Q_a = np.zeros((N + 1, N + 1))
    Q_b = np.zeros((N + 1, N + 1))
    best_cost = init_cost; best_route = init_route[:]

    for episode in range(episodes):
        eps = 1.0 - episode / episodes      # e1: linear decay (Wang et al.)

        cur_t = 0.0; cur_node = 0
        unvisited = set(range(1, N + 1))
        route = []; travel_cost = 0.0

        while unvisited:
            cands = feasible_nbrs(cur_node, cur_t, unvisited, l, t)

            # ── dead-end: fall back to EDF greedy to finish the tour ──────────
            if not cands:
                extra, completion = greedy_complete(
                    cur_node, cur_t, unvisited, e, l, d, t)
                if completion is not None:
                    travel_cost += extra
                    full_route = route + completion
                    # penalise the state that led to the dead-end
                    prev = route[-1] if route else 0
                    update_Q(prev, cur_node, -10.0, 0, Q_a, Q_b)
                    if travel_cost < best_cost:
                        best_cost = travel_cost
                        best_route = full_route[:]
                break

            action = next_action(cur_node, cands, Q_a, Q_b, eps)
            r = reward_r1(t[cur_node][action])

            new_t = arrive(cur_t, cur_node, action, e, t) + d[action]
            new_unvisited = unvisited - {action}

            if new_unvisited:
                nc = feasible_nbrs(action, new_t, new_unvisited, l, t)
                a_t1 = next_action(action, nc, Q_a, Q_b, eps) if nc else 0
            else:
                a_t1 = 0

            update_Q(cur_node, action, r, a_t1, Q_a, Q_b)

            travel_cost += t[cur_node][action]
            route.append(action)
            cur_t = new_t; cur_node = action; unvisited = new_unvisited

        else:
            # All nodes visited via Q-learning (no greedy fallback needed)
            travel_cost += t[cur_node][0]
            dc = feasible_nbrs(0, 0.0, set(range(1, N+1)), l, t)
            a_t1 = next_action(0, dc, Q_a, Q_b, eps) if dc else 0
            update_Q(cur_node, 0, reward_r1(t[cur_node][0]), a_t1, Q_a, Q_b)
            if travel_cost < best_cost:
                best_cost = travel_cost; best_route = route[:]

    return best_cost, best_route


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    data = sys.stdin.read().replace('\r', '').split()
    idx = 0
    N = int(data[idx]); idx += 1

    e = [0] * (N + 1); l = [10**9] * (N + 1); d = [0] * (N + 1)
    for i in range(1, N + 1):
        e[i] = int(data[idx]); idx += 1
        l[i] = int(data[idx]); idx += 1
        d[i] = int(data[idx]); idx += 1

    t = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            t[i][j] = int(data[idx]); idx += 1

    # Phase 1 – greedy initialisation
    best_cost, best_route = find_initial_solution(N, e, l, d, t)
    if best_route is None:
        print(-1.0); return

    # Phase 2 – Double Q-Learning
    episodes = (3000 if N <= 10 else 1500 if N <= 50 else
                600  if N <= 200 else 300  if N <= 500 else
                150  if N <= 800 else 80)
    best_cost, best_route = run_double_q_learning(
        N, e, l, d, t, episodes, best_cost, best_route)

    # Phase 3 – local search (budget-limited for large N)
    if N <= 500:
        c2, r2 = two_opt(best_route, e, l, d, t, max_iter=max(1, 200 - N // 3))
        if c2 < best_cost:
            best_cost, best_route = c2, r2

    if N <= 300:
        co, ro = or_opt(best_route, e, l, d, t)
        if co < best_cost:
            best_cost, best_route = co, ro

    print(float(best_cost))


if __name__ == "__main__":
    main()
