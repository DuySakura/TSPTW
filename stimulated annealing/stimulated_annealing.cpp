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

double cal_cost(const vector<int> &route, const double &penalty) {
    double cost = t[0][route[0]+1];
    double cur_time = max(t[0][route[0]+1], e[route[0]]);
    double total_penalty = max(0.0, cur_time - l[route[0]]);

    for (int i = 1; i < n; ++i) {
        cost += t[route[i-1]+1][route[i]+1];
        cur_time = max(cur_time + d[route[i-1]] + t[route[i-1]+1][route[i]+1], e[route[i]]);
        total_penalty += max(0.0, cur_time - l[route[i]]);
    }

    return cost + t[route[n-1]+1][0] + penalty * total_penalty;
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
    double T_start = 10000.0;       
    double T_min = 1e-3;
    double T = T_start;
    double cooling_rate = 0.99;
    int markov_chain = max(100, 10 * n);

    double penalty_start = 0.1;
    double penalty_max = 5000.0;
    double penalty = penalty_start;

    mt19937 gen(18);
    uniform_int_distribution<> random_idx(0, n - 1);
    uniform_int_distribution<> random_op(0, 2);
    uniform_real_distribution<> random_prob(0.0, 1.0);

    vector<int> best_route(n);
    for (int i = 0; i < n; ++i) best_route[i] = i;
    sort(best_route.begin(), best_route.end(), [] (const int &a, const int &b) {
        return l[a] < l[b];
    });
    double best_cost = cal_cost(best_route, penalty);

    vector<int> current_route = best_route;
    double current_cost = best_cost;

    while (T > T_min) {
        double progress = log(T_start / T) / log(T_start / T_min);
        penalty = penalty_start + (penalty_max - penalty_start) * (progress * progress);
        current_cost = cal_cost(current_route, penalty);

        for (int step = 0; step < markov_chain; ++step) {
            int i = random_idx(gen);
            int j = random_idx(gen);
            while (i == j) {
                i = random_idx(gen);
                j = random_idx(gen);
            }

            int op = random_op(gen);
            vector<int> route = current_route;
            if (op == 0) swap(route[i], route[j]);
            else if (op == 1) std::reverse(route.begin() + min(i, j), route.begin() + max(i, j) + 1);
            else {
                int val = route[i];
                int pos = (j > i) ? j - 1 : j;
                route.erase(route.begin() + i);
                route.insert(route.begin() + pos, val);
            }
            double cost = cal_cost(route, penalty);
            double delta = cost - current_cost;

            if (delta < 0 || random_prob(gen) < exp(-delta / T)) {
                current_route = route;
                current_cost = cost;

                if (is_feasible(current_route)) {
                    if (current_cost < best_cost) {
                        best_route = current_route;
                        best_cost = current_cost;
                    }
                }
            }
        }
        T *= cooling_rate; 
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
