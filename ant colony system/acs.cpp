#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<double>> t;
vector<double> e, l, d;

// --- CÁC THAM SỐ CỦA THUẬT TOÁN ACS-TSPTW (Được lấy trực tiếp từ Bài báo) ---
const int m_ants = 3;               // Số lượng kiến mỗi vòng lặp
const double q0 = 0.99;             // Tỷ lệ khai thác (Exploitation)
const double theta_param = 0.1;     // Hệ số bay hơi pheromone toàn cục
const double omega = 0.1;           // Hệ số bay hơi pheromone cục bộ
const double beta_param = 0.5;      // Trọng số của heuristic g_ij (Slack time)
const double gamma_param = 3.0;     // Trọng số của heuristic h_ij (Waiting time)
const double delta_param = 0.05;    // Độ dốc hàm sigmoid của g_ij
const double lambda_param = 0.05;   // Độ dốc hàm sigmoid của h_ij

void init() {
    t.assign(n + 1, vector<double>(n + 1));
    e.resize(n);
    l.resize(n);
    d.resize(n);
}

double cal_cost(const vector<int> &route) {
    double cost = t[0][route[0]];
    double cur_time = max(t[0][route[0]], e[route[0]-1]);

    for (int i = 1; i < n; ++i) {
        cost += t[route[i-1]][route[i]];
        cur_time = max(cur_time + d[route[i-1]-1] + t[route[i-1]][route[i]], e[route[i]-1]);

        if (cur_time > l[route[i]-1]) return 2e18;
    }

    return cost + t[route[n-1]][0];
}

double solve() {
    int max_iterations = 50000;
    
    // 1. Khởi tạo Pheromone ban đầu (tau_0) bằng Nearest Neighbor
    double L_NN = 0;
    vector<bool> visited(n + 1, false);
    int curr = 0;
    double cur_time = 0;
    for (int step = 0; step < n; ++step) {
        int best_next = -1;
        double min_dist = 1e9;
        for (int j = 1; j <= n; ++j) {
            if (!visited[j] && t[curr][j] < min_dist) {
                min_dist = t[curr][j];
                best_next = j;
            }
        }
        L_NN += min_dist;
        visited[best_next] = true;
        curr = best_next;
    }
    L_NN += t[curr][0];
    
    double tau_0 = 1.0 / (n * L_NN);
    vector<vector<double>> tau(n + 1, vector<double>(n + 1, tau_0));

    // Biến lưu kỷ lục toàn cục
    vector<int> global_best_route;
    double global_best_cost = 2e18;

    mt19937 gen(18);
    uniform_real_distribution<> random_prob(0.0, 1.0);

    for (int iter = 1; iter <= max_iterations; ++iter) {
        vector<int> iteration_best_route;
        double iteration_best_cost = 2e18;

        // Cho từng con kiến xây dựng hành trình
        for (int k = 0; k < m_ants; ++k) {
            vector<int> route;
            vector<bool> unvisited(n + 1, true);
            unvisited[0] = false;
            int current_node = 0;
            double current_time = 0;
            bool is_feasible = true;

            for (int step = 0; step < n; ++step) {
                vector<int> N_i;
                vector<double> G_val, H_val;
                double sum_G = 0, sum_H = 0;
                int count_G = 0, count_H = 0;

                // Tính toán G_ij và H_ij cho tất cả đỉnh chưa thăm
                for (int j = 1; j <= n; ++j) {
                    if (unvisited[j]) {
                        double service_time = (current_node == 0) ? 0 : d[current_node - 1];
                        double t_j = current_time + service_time + t[current_node][j];
                        
                        double G_ij = l[j-1] - t_j;
                        double H_ij = e[j-1] - t_j;
                        
                        N_i.push_back(j);
                        G_val.push_back(G_ij);
                        H_val.push_back(H_ij);
                        
                        if (G_ij >= 0) { sum_G += G_ij; count_G++; }
                        if (H_ij > 0) { sum_H += H_ij; count_H++; }
                    }
                }

                if (count_G == 0) {
                    // Tất cả các đỉnh đều vi phạm thời gian -> Kiến bỏ cuộc (Theo bài báo)
                    is_feasible = false;
                    break;
                }

                double mu = (count_G > 0) ? (sum_G / count_G) : 0;
                double nu = (count_H > 0) ? (sum_H / count_H) : 0;

                vector<double> prob(N_i.size(), 0.0);
                double prob_sum = 0;
                int best_j = -1;
                double max_eval = -1;

                // Tính toán g_ij, h_ij và biểu thức trạng thái
                for (size_t idx = 0; idx < N_i.size(); ++idx) {
                    int j = N_i[idx];
                    double g_ij = 0, h_ij = 1.0;

                    if (G_val[idx] >= 0) {
                        g_ij = 1.0 / (1.0 + exp(delta_param * (G_val[idx] - mu)));
                    }
                    if (H_val[idx] > 0) {
                        h_ij = 1.0 / (1.0 + exp(lambda_param * (H_val[idx] - nu)));
                    }

                    double eval = tau[current_node][j] * pow(g_ij, beta_param) * pow(h_ij, gamma_param);
                    prob[idx] = eval;
                    prob_sum += eval;

                    if (eval > max_eval) {
                        max_eval = eval;
                        best_j = j;
                    }
                }

                // Chọn đỉnh tiếp theo (Exploitation vs Exploration)
                int next_node = -1;
                double q = random_prob(gen);
                
                if (q <= q0) {
                    next_node = best_j; // Khai thác
                } else {
                    // Khám phá bằng Vòng quay Roulette
                    double rand_val = random_prob(gen) * prob_sum;
                    double cumulative = 0;
                    for (size_t idx = 0; idx < N_i.size(); ++idx) {
                        cumulative += prob[idx];
                        if (cumulative >= rand_val) {
                            next_node = N_i[idx];
                            break;
                        }
                    }
                    if (next_node == -1) next_node = N_i.back();
                }

                // Cập nhật trạng thái
                route.push_back(next_node);
                unvisited[next_node] = false;
                
                double s_time = (current_node == 0) ? 0 : d[current_node-1];
                current_time = max(current_time + s_time + t[current_node][next_node], e[next_node-1]);
                
                // Cập nhật Pheromone cục bộ (Local Updating Rule)
                tau[current_node][next_node] = (1.0 - omega) * tau[current_node][next_node] + omega * tau_0;
                current_node = next_node;
            }

            if (is_feasible) {
                double cost = cal_cost(route);
                if (cost < iteration_best_cost) {
                    iteration_best_cost = cost;
                    iteration_best_route = route;
                }
            }
        } // Kết thúc 1 vòng lặp của m kiến

        // Cập nhật Pheromone toàn cục (Global Updating Rule)
        if (iteration_best_cost < global_best_cost) {
            global_best_cost = iteration_best_cost;
            global_best_route = iteration_best_route;
        }

        // Chỉ cập nhật pheromone cho hành trình TỐT NHẤT TOÀN CỤC (Theo quy tắc ACS)
        if (global_best_cost != 2e18) {
            int prev = 0;
            for (int i = 0; i < n; ++i) {
                int curr = global_best_route[i];
                tau[prev][curr] = (1.0 - theta_param) * tau[prev][curr] + theta_param * (1.0 / global_best_cost);
                prev = curr;
            }
            tau[prev][0] = (1.0 - theta_param) * tau[prev][0] + theta_param * (1.0 / global_best_cost);
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
