#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<int>> t, travel_time, arrival_time, route;
vector<int> e, l, d;

void init() {
    t.assign(n + 1, vector<int>(n + 1));
    travel_time.assign(1 << n, vector<int>(n, INT_MAX));
    arrival_time.assign(1 << n, vector<int>(n, -1));
    route.assign(1 << n, vector<int>(n));
    e.resize(n);
    l.resize(n); 
    d.resize(n);
}

void solve() {
    for (unsigned long long mask = 1; mask < (1 << n); ++mask)
    for (int i = 0; i < n; ++i) {
        if ((mask & (1 << i)) == 0) continue;

        if (mask == (1 << i)) {
            if (t[0][i+1] > l[i]) continue;

            travel_time[1 << i][i] = t[0][i+1];
            arrival_time[1 << i][i] = max(t[0][i+1], e[i]);
            route[1 << i][i] = 0;
            continue;
        }

        for (int j = 0; j < n; ++j) {
            unsigned long long prev_mask = mask ^ (1 << i);

            if (travel_time[prev_mask][j] == INT_MAX || arrival_time[prev_mask][j] + d[j] + t[j+1][i+1] > l[i]) continue;

            if (travel_time[mask][i] > travel_time[prev_mask][j] + t[j+1][i+1]) {
                travel_time[mask][i] = travel_time[prev_mask][j] + t[j+1][i+1];
                arrival_time[mask][i] = max(arrival_time[prev_mask][j] + d[j] + t[j+1][i+1], e[i]);
                route[mask][i] = j;
            }
        }
    }
}

int main() {
    cin >> n;
    init();
    for (int i = 0; i < n; ++i) cin >> e[i] >> l[i] >> d[i];
    for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j) cin >> t[i][j];

    solve();

    int res = INT_MAX;
    for (int i = 0; i < n; ++i) {
        if (travel_time[(1 << n) - 1][i] != INT_MAX) res = min(res, travel_time[(1 << n) - 1][i] + t[i+1][0]);
    }
    cout << res << endl;
}
