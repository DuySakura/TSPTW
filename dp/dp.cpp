#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Sử dụng 1 tỷ làm giá trị Vô cùng để tránh lỗi tràn số nguyên (Overflow)
const int INF = 1e9; 

int n;
vector<vector<int>> t;
vector<int> e, l, d;

// 3 Bảng Quy hoạch động
vector<vector<int>> travel_time;
vector<vector<int>> arrival_time;
vector<vector<int>> trace_node;

void init() {
    int N = n + 1; // Tổng số đỉnh = Khách hàng + 1 Kho
    t.assign(N, vector<int>(N, 0));
    e.assign(N, 0);
    l.assign(N, 0);
    d.assign(N, 0);

    int max_mask = 1 << N;
    travel_time.assign(max_mask, vector<int>(N, INF));
    arrival_time.assign(max_mask, vector<int>(N, INF));
    trace_node.assign(max_mask, vector<int>(N, -1));
}

void solve() {
    int N = n + 1;
    int max_mask = 1 << N;

    // Trạng thái cơ sở tại Kho (Đỉnh 0)
    travel_time[1][0] = 0;
    arrival_time[1][0] = e[0];

    // 1. CHẠY QUY HOẠCH ĐỘNG (Xây bảng DP)
    for (int mask = 1; mask < max_mask; ++mask) {
        for (int i = 0; i < N; ++i) {
            // Bỏ qua nếu đỉnh i chưa được thăm trong tập hợp mask
            if (!(mask & (1 << i))) continue;

            int prev_mask = mask ^ (1 << i);
            if (prev_mask == 0) continue; // Bỏ qua base case

            for (int j = 0; j < N; ++j) {
                if (!(prev_mask & (1 << j))) continue;
                if (travel_time[prev_mask][j] == INF) continue;

                // Tính thời gian thực tế lúc đến đỉnh i
                int arr = arrival_time[prev_mask][j] + d[j] + t[j][i];
                int start = max(arr, e[i]); // Nếu đến sớm thì phải chờ đến e[i]

                // Kiểm tra Time Window (Chỉ đi tiếp nếu không bị trễ)
                if (start <= l[i]) {
                    // Cập nhật mục tiêu: Tổng thời gian di chuyển (không tính thời gian chờ)
                    int new_travel = travel_time[prev_mask][j] + t[j][i];

                    if (new_travel < travel_time[mask][i]) {
                        travel_time[mask][i] = new_travel;
                        arrival_time[mask][i] = start; // Lưu lại thời gian đến cho bước sau
                        trace_node[mask][i] = j;       // Lưu vết đỉnh j để backtracking
                    }
                }
            }
        }
    }

    // 2. CHỐT KẾT QUẢ TẠI ĐÍCH (Quay về đỉnh 0)
    int full_mask = max_mask - 1;
    int best_travel = INF;
    int best_last = -1;

    for (int j = 1; j < N; ++j) {
        if (travel_time[full_mask][j] != INF) {
            // Chặng cuối từ j về 0 (Kho không giới hạn l[0], luôn hợp lệ)
            int total_travel = travel_time[full_mask][j] + t[j][0];
            
            if (total_travel < best_travel) {
                best_travel = total_travel;
                best_last = j;
            }
        }
    }

    if (best_travel == INF) {
        cout << "Khong tim thay nghiem (Infeasible)\n";
        return;
    }

    // 3. TRUY VẾT LỘ TRÌNH (BACKTRACKING)
    vector<int> path;
    int curr_mask = full_mask;
    int curr_node = best_last;

    while (curr_node != -1) {
        path.push_back(curr_node);
        int prev = trace_node[curr_mask][curr_node];
        curr_mask ^= (1 << curr_node);
        curr_node = prev;
    }

    // Đảo ngược vector vì quá trình truy vết đi từ đích về nguồn
    reverse(path.begin(), path.end());
    path.push_back(0); // Nối đỉnh kho vào cuối

    // IN KẾT QUẢ
    cout << n << endl;
    for (size_t i = 1; i < path.size() - 1; ++i) {
        cout << path[i] << (i + 1 == path.size() ? "" : " ");
    }
    cout << "\n";
}

int main() {
    // Tối ưu hóa luồng I/O trong C++ giúp đọc file nhanh hơn
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n)) return 0;

    init();

    // Thiết lập thông số cho đỉnh 0 (Kho)
    e[0] = 0;
    l[0] = INF; // Đóng cửa rất muộn
    d[0] = 0;   // Không tính thời gian phục vụ tại kho

    // Đọc thông số n khách hàng (Lưu vào vị trí 1 đến n)
    for (int i = 1; i <= n; ++i) {
        cin >> e[i] >> l[i] >> d[i];
    }

    // Đọc ma trận thời gian đi lại (Kích thước n+1 x n+1)
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            cin >> t[i][j];
        }
    }

    solve();

    return 0;
}
