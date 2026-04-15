from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve(n, e, l, d, t):
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def travel_time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)

        return t[from_node][to_node]
    
    travel_time_callback_idx = routing.RegisterTransitCallback(travel_time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(travel_time_callback_idx)

    def arrival_time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)

        return d[from_node] + t[from_node][to_node]
    
    arrival_time_callback_idx = routing.RegisterTransitCallback(arrival_time_callback)
    
    routing.AddDimension(
        arrival_time_callback_idx,
        max(l),
        max(l) + 2 * max(max(row) for row in t),
        True,
        'time'
    )

    time_dimension = routing.GetDimensionOrDie('time')

    for i in range(n):
        time_dimension.CumulVar(manager.NodeToIndex(i)).SetRange(e[i], l[i])

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 5

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print(solution.ObjectiveValue())
        # print(n - 1)

        # idx = routing.Start(0)
        
        # while True:
        #     idx = solution.Value(routing.NextVar(idx))

        #     if routing.IsEnd(idx):
        #         break

        #     print(manager.IndexToNode(idx), end=' ')

        # print()
    
    else:
        print(-1)


if __name__ == "__main__":
    n = int(input())

    tmp = [list(map(int, input().split())) for _ in range(n)]
    t = [list(map(int, input().split())) for _ in range(n + 1)]
    e = [0] + [row[0] for row in tmp]
    l = [max(row[1] for row in tmp) + max(row[0] for row in t)] + [row[1] for row in tmp]
    d = [0] + [row[2] for row in tmp]

    solve(n + 1, e, l, d, t)
