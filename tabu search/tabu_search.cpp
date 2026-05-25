#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<double>> t;
vector<double> e, l, d;

void init() {
    t.assign(n + 1, vector<double>(n + 1));
    e.resize(n);
    l.resize(n); 
    d.resize(n);
}

double cal_cost(const vector<int> &route, const double &alpha) {
    double cost = t[0][route[0]+1];
    double cur_time = max(t[0][route[0]+1], e[route[0]]);
    double total_penalty = max(0.0, cur_time - l[route[0]]);

    for (int i = 1; i < n; ++i) {
        cost += t[route[i-1]+1][route[i]+1];
        cur_time = max(cur_time + d[route[i-1]] + t[route[i-1]+1][route[i]+1], e[route[i]]);
        total_penalty += max(0.0, cur_time - l[route[i]]);
    }

    return cost + t[route[n-1]+1][0] + alpha * total_penalty;
}

bool is_feasible(const vector<int> &route) {
    double cur_time = max(t[0][route[0]+1], e[route[0]]);

    for (int i = 0; i < n - 1; ++i) {
        if (cur_time > l[route[i]]) return false;

        cur_time = max(cur_time + d[route[i]] + t[route[i]+1][route[i+1]+1], e[route[i+1]]);
    }

    return cur_time <= l[route[n-1]];
}

double solve() {
    int no_improve = 0;
    int max_no_improve = 1000 * n;
    auto start_time = chrono::steady_clock::now();

    double alpha = 10;
    double gamma = 1.2;
    int feasible_count = 0;
    int check_period = 50;

    mt19937 gen(18);
    uniform_int_distribution<> random_idx(0, n - 1);
    uniform_int_distribution<> random_tenure(5, 15);

    vector<int> best_route(n);
    for (int i = 0; i < n; ++i) best_route[i] = i;
    sort(best_route.begin(), best_route.end(), [] (const int &a, const int &b) {
        return l[a] < l[b];
    });
    double best_cost = cal_cost(best_route, alpha);

    vector<int> current_route = best_route;
    double current_cost = best_cost;

    vector<vector<int>> tabu_list(n, vector<int>(n, 0));
    int neighborhood_size = min(200, (n * (n - 1)) / 2);

    for (int iter = 1; ; ++iter) {
        if (no_improve > max_no_improve) break;
        
        auto current_time = chrono::steady_clock::now();
        double elapsed = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
        if (elapsed > 55) break;

        if (iter % check_period == 0) {
            if (feasible_count >= check_period) alpha = max(0.1, alpha / gamma);
            else alpha = min(100000.0, alpha * gamma);

            current_cost = cal_cost(current_route, alpha);
            feasible_count = 0;
        }

        vector<int> best_neighbor_route;
        double best_neighbor_cost = 2e18;
        int best_swap_u = -1, best_swap_v = -1;

        for (int step = 0; step < neighborhood_size; ++step) {
            int i = random_idx(gen);
            int j = random_idx(gen);
            while (i == j) {
                i = random_idx(gen);
                j = random_idx(gen);
            }

            vector<int> route = current_route;
            swap(route[i], route[j]);
            double cost = cal_cost(route, alpha);

            int u = route[i];
            int v = route[j];

            bool is_tabu = (tabu_list[u][v] >= iter);
            bool meets_aspiration = false;
            if (cost < best_cost && is_feasible(route)) {
                meets_aspiration = true;
            }

            if (!is_tabu || meets_aspiration) {
                if (cost < best_neighbor_cost) {
                    best_neighbor_route = route;
                    best_neighbor_cost = cost;
                    best_swap_u = u;
                    best_swap_v = v;
                }
            }
        }

        if (best_swap_u == -1) continue; 

        current_route = best_neighbor_route;
        current_cost = best_neighbor_cost;

        int tenure = random_tenure(gen);
        tabu_list[best_swap_u][best_swap_v] = iter + tenure;
        tabu_list[best_swap_v][best_swap_u] = iter + tenure;

        bool improve = false;
        bool feasible = is_feasible(best_neighbor_route);

        if (feasible) ++feasible_count;
        if (best_neighbor_cost < current_cost) {
            current_route = best_neighbor_route;
            current_cost = best_neighbor_cost;

            if (feasible) {
                best_route = current_route;
                best_cost = current_cost;
                improve = true;
            }
        }

        if (improve) no_improve = 0;
        else ++no_improve;
    }

    return best_cost;
}

int main() {
    cin >> n;
    init();
    for (int i = 0; i < n; ++i) cin >> e[i] >> l[i] >> d[i];
    for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j) cin >> t[i][j];

    cout << solve() << endl;
}
