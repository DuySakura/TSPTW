import numpy as np
from ortools.linear_solver import pywraplp


def solve(n, e, l, d, t):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return
    
    x = np.array([
        [solver.BoolVar(f'x_{i}_{j}') for j in range(n)]
        for i in range(n)
    ])

    w = []
    for i in range(n):
        w.append(solver.IntVar(int(e[i]), int(l[i]), f'w_{i}'))

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

    solver.Minimize(np.sum(t * x))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        # print(solver.Objective().Value())
        print(n - 1)

        prev = 0
        for _ in range(n - 1):
            for i in range(n):
                if x[prev, i].solution_value() == True:
                    print(i, end=' ')
                    prev = i
                    break
        print()
    
    else:
        print("Không tìm thấy nghiệm tối ưu")

    
if __name__ == "__main__":
    n = int(input())

    tmp = np.array([list(map(int, input().split())) for _ in range(n)])
    t = np.array([list(map(int, input().split())) for _ in range(n + 1)])
    e = np.insert(tmp[:, 0], 0, 0)
    l = np.insert(tmp[:, 1], 0, np.max(tmp[:, 1]) + np.max(t[:, 0]))
    d = np.insert(tmp[:, 2], 0, 0)

    solve(n + 1, e, l, d, t)
