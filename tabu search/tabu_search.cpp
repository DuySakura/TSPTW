#include <bits/stdc++.h>
using namespace std;

int n, pen = 1e6;
vector<vector<int>> t;
vector<int> e, l, d;

void init() {
    t.assign(n + 1, vector<int>(n + 1));
    e.resize(n);
    l.resize(n); 
    d.resize(n);
}

long long cal_cost(const vector<int> &route) {
    long long cost = t[0][route[0]+1], cur_time = max(t[0][route[0]+1], e[route[0]]) + d[route[0]];

    for (int i = 1; i < n; ++i) {
        cost += t[route[i-1]+1][route[i]+1];
        if (cur_time > l[route[i]]) cost += pen;
        cur_time = max(cur_time + t[route[i-1]+1][route[i]+1], (long long)e[route[i]]) + d[route[i]];
    }

    return cost + t[route[n-1]+1][0];
}

long long solve(int max_iterations = 1000) {
    vector<int> current_route(n);
    for (int i = 0; i < n; ++i) current_route[i] = i;
    sort(current_route.begin(), current_route.end(), [] (const int &a, const int &b) {
        return l[a] < l[b];
    });
    
    long long current_cost = cal_cost(current_route);
    
    vector<int> global_best_route = current_route;
    long long global_best_cost = current_cost;

    vector<vector<int>> tabu_list(n, vector<int>(n, 0));
    int tabu_tenure = 10;

    mt19937 gen(18);
    uniform_int_distribution<> random_node(0, n - 1);

    for (int iter = 1; iter <= max_iterations; ++iter) {
        long long best_neighbor_cost = 2e18;
        vector<int> best_neighbor_route;
        int best_swap_u = -1, best_swap_v = -1;

        for (int step = 0; step < 50; ++step) {
            int i = random_node(gen);
            int j = random_node(gen);
            if (i == j) continue;

            vector<int> neighbor_route = current_route;
            swap(neighbor_route[i], neighbor_route[j]);
            long long neighbor_cost = cal_cost(neighbor_route);

            int u = neighbor_route[i];
            int v = neighbor_route[j];

            bool is_tabu = (tabu_list[u][v] >= iter);
            bool meets_aspiration = (neighbor_cost < global_best_cost);

            if (!is_tabu || meets_aspiration) {
                if (neighbor_cost < best_neighbor_cost) {
                    best_neighbor_cost = neighbor_cost;
                    best_neighbor_route = neighbor_route;
                    best_swap_u = u;
                    best_swap_v = v;
                }
            }
        }

        if (best_swap_u == -1) continue;

        current_route = best_neighbor_route;
        current_cost = best_neighbor_cost;

        tabu_list[best_swap_u][best_swap_v] = iter + tabu_tenure;
        tabu_list[best_swap_v][best_swap_u] = iter + tabu_tenure;

        if (current_cost < global_best_cost) {
            global_best_cost = current_cost;
            global_best_route = current_route;
        }
    }

    return global_best_cost;
}

int main() {
    cin >> n;
    init();
    for (int i = 0; i < n; ++i) cin >> e[i] >> l[i] >> d[i];
    for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j) cin >> t[i][j];

    cout << solve() << endl;
}
