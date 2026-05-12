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

long long solve(int max_iterations = 100) {
    vector<int> best_route(n);
    for (int i = 0; i < n; ++i) best_route[i] = i;
    sort(best_route.begin(), best_route.end(), [] (const int &a, const int &b) {
        return l[a] < l[b];
    });
    int best_cost = cal_cost(best_route);

    mt19937 gen(18);
    uniform_int_distribution<> random(0, n - 1);

    while (max_iterations--) {
        int i = random(gen);
        int j = random(gen);
        vector<int> route = best_route;
        swap(route[i], route[j]);
        int cost = cal_cost(route);

        if (cost < best_cost) {
            best_route = route;
            best_cost = cost;
            continue;
        }

        i = random(gen);
        j = random(gen);
        route = best_route;
        int val = route[i];
        route.erase(route.begin() + i);
        route.insert(route.begin() + j, val);
        cost = cal_cost(route);

        if (cost < best_cost) {
            best_route = route;
            best_cost = cost;
        }
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
