import sys
import numpy as np
from ortools.linear_solver import pywraplp


def solve(n, e, l, d, t, objective):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return
    
    solver.SetTimeLimit(55000)
    
    x = np.array([
        [solver.BoolVar(f'x_{i}_{j}') for j in range(n)]
        for i in range(n)
    ])

    w = []
    for i in range(n):
        w.append(solver.NumVar(float(e[i]), float(l[i]), f'w_{i}'))

    for i in range(n):
        solver.Add(np.sum(x[i, :]) == 1)
        solver.Add(np.sum(x[:, i]) == 1)
        solver.Add(x[i, i] == 0)

    M = np.max(l) - np.min(e) + np.max(d) + np.max(t)
    for i in range(n):
        for j in range(n):
            if i != j and i != 0 and j != 0:
                solver.Add(w[i] + d[i] + t[i][j] - M * (1 - x[i, j]) <= w[j])

    for i in range(n):
        solver.Add(w[i] >= e[i])
        solver.Add(w[i] <= l[i])

    if objective == 'makespan':
        makespan_var = solver.NumVar(0, float(M), 'makespan')
        for i in range(1, n):
            solver.Add(makespan_var >= w[i] + d[i] + t[i][0] - M * (1 - x[i, 0]))

        solver.Minimize(makespan_var)

    elif objective == 'travel_time':
        solver.Minimize(np.sum(t * x))

    else:
        raise ValueError(f"Hàm mục tiêu không hợp lệ: {objective}")

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(solver.Objective().Value())
    
    else:
        print(-1)

    
if __name__ == "__main__":
    objective = sys.argv[1] if len(sys.argv) > 1 else "makespan"
    
    n = int(input())
    tmp = np.array([list(map(float, input().split())) for _ in range(n)])
    t = np.array([list(map(float, input().split())) for _ in range(n + 1)])
    e = np.insert(tmp[:, 0], 0, 0)
    l = np.insert(tmp[:, 1], 0, np.max(tmp[:, 1]) + np.max(t[:, 0]))
    d = np.insert(tmp[:, 2], 0, 0)

    solve(n + 1, e, l, d, t)
