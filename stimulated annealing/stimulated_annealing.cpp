#include <bits/stdc++.h>
using namespace std;

int n, pen = 1e6;
vector<vector<double>> t;
vector<double> e, l, d;

void init() {
    t.assign(n + 1, vector<double>(n + 1));
    e.resize(n);
    l.resize(n); 
    d.resize(n);
}

double cal_cost(const vector<int> &route) {
    double cost = t[0][route[0]+1], cur_time = max(t[0][route[0]+1], e[route[0]]) + d[route[0]];

    for (int i = 1; i < n; ++i) {
        cost += t[route[i-1]+1][route[i]+1];
        if (cur_time > l[route[i]]) cost += pen;
        cur_time = max(cur_time + t[route[i-1]+1][route[i]+1], e[route[i]]) + d[route[i]];
    }

    return cost + t[route[n-1]+1][0];
}

double solve(int max_iterations = 1000) {
    vector<int> current_route(n);
    for (int i = 0; i < n; ++i) current_route[i] = i;
    sort(current_route.begin(), current_route.end(), [] (const int &a, const int &b) {
        return l[a] < l[b];
    });
    
    double current_cost = cal_cost(current_route);
    
    vector<int> global_best_route = current_route;
    double global_best_cost = current_cost;

    double T = 10000.0;       
    double T_min = 1e-3;      
    double alpha = 0.99;      
    int markov_chain = 100;

    mt19937 gen(18);
    uniform_int_distribution<> random_node(0, n - 1);
    uniform_real_distribution<> random_prob(0.0, 1.0);

    while (T > T_min) {
        for (int step = 0; step < markov_chain; ++step) {
            int i = random_node(gen);
            int j = random_node(gen);
            if (i == j) continue;

            vector<int> neighbor_route = current_route;
            swap(neighbor_route[i], neighbor_route[j]);
            double neighbor_cost = cal_cost(neighbor_route);
            double delta = neighbor_cost - current_cost;

            if (delta < 0) {
                current_route = neighbor_route;
                current_cost = neighbor_cost;

                if (current_cost < global_best_cost) {
                    global_best_cost = current_cost;
                    global_best_route = current_route;
                }
            } 
            else {
                double p = exp(-delta / T);
                if (random_prob(gen) < p) {
                    current_route = neighbor_route;
                    current_cost = neighbor_cost;
                }
            }
        }
        
        T *= alpha; 
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
