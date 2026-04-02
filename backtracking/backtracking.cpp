#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<int>> t;
vector<int> e, l, d;
int t_min = INT_MAX;
vector<bool> visited;
vector<int> route, s;
int res = INT_MAX;

void init() {
    t.assign(n + 1, vector<int>(n + 1));
    e.resize(n);
    l.resize(n); 
    d.resize(n);
    visited.assign(n, false);
}

void solve(const int &prev, int k, int cur_time, int travel_time) {
    if (k == n) {
        if (res > travel_time + t[prev][0]) {
            s = route;
            res = travel_time + t[prev][0];
        }
        return;
    }

    for (int i = 1; i <= n; ++i) {
        if (visited[i-1]) continue;
        int arrival_time = cur_time + t[prev][i];
        if (arrival_time > l[i-1]) return;
        if (res <= travel_time + t[prev][i] + (n - k - 1) * t_min) continue;
        int serve_time = max(arrival_time, e[i-1]);
        visited[i-1] = true;
        route.push_back(i);
        solve(i, k + 1, serve_time + d[i-1], travel_time + t[prev][i]);
        visited[i-1] = false;
        route.pop_back();
    }
}

int main() {
    cin >> n;
    init();
    for (int i = 0; i < n; ++i) cin >> e[i] >> l[i] >> d[i];
    for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j) {
        cin >> t[i][j];
        if (i != j) t_min = min(t_min, t[i][j]);
    }

    solve(0, 0, 0, 0);
    // cout << n << endl;
    // for (const auto &x : s) cout << x << ' ';
    cout << res << endl;
}
