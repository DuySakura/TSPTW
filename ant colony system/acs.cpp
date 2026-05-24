#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<double>> t;
vector<double> e, l, d;

// --- CÁC THAM SỐ CỦA THUẬT TOÁN ACS-TSPTW ---
const int m_ants = 3;               
const double q0 = 0.99;             
const double theta_param = 0.1;     
const double omega = 0.1;           
const double beta_param = 0.5;      
const double gamma_param = 3.0;     
const double delta_param = 0.05;    
const double lambda_param = 0.05;   

void init() {
    t.assign(n + 1, vector<double>(n + 1));
    e.resize(n);
    l.resize(n);
    d.resize(n);
}

// 1. Hàm tính cost TÍCH HỢP HÀM PHẠT
double cal_cost(const vector<int> &route, double penalty) {
    if (route.empty()) return 2e18;
    
    int first_node = route[0];
    double cost = t[0][first_node];
    double cur_time = max(t[0][first_node], e[first_node - 1]);
    double total_penalty = max(0.0, cur_time - l[first_node - 1]);

    for (int i = 1; i < n; ++i) {
        int prev = route[i-1];
        int curr = route[i];
        
        cost += t[prev][curr];
        cur_time = max(cur_time + d[prev - 1] + t[prev][curr], e[curr - 1]);
        
        total_penalty += max(0.0, cur_time - l[curr - 1]); // Cộng dồn độ trễ
    }

    cost += t[route[n-1]][0]; // Cộng thêm quãng đường quay về kho
    
    return cost + penalty * total_penalty;
}

// 2. Hàm kiểm tra nghiệm khả thi tuyệt đối
bool is_feasible(const vector<int> &route) {
    if (route.empty()) return false;
    
    int first_node = route[0];
    double cur_time = max(t[0][first_node], e[first_node - 1]);
    if (cur_time > l[first_node - 1]) return false;

    for (int i = 1; i < n; ++i) {
        int prev = route[i-1];
        int curr = route[i];
        cur_time = max(cur_time + d[prev - 1] + t[prev][curr], e[curr - 1]);
        if (cur_time > l[curr - 1]) return false;
    }
    return true;
}

double solve() {
    int no_improve = 0;
    int max_no_improve = 50 * n;
    auto start_time = chrono::steady_clock::now();

    double penalty = 100.0; // Hệ số phạt cố định đủ lớn để ép kiến tìm đường đúng
    
    // Khởi tạo Pheromone ban đầu (tau_0) bằng Nearest Neighbor
    double L_NN = 0;
    vector<bool> visited(n + 1, false);
    int curr = 0;
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
    double global_best_cost = 2e18;             // Chứa nghiệm phạt (để rải Pheromone)
    double true_best_feasible_cost = 2e18;      // Chứa nghiệm hợp lệ (để in kết quả)

    mt19937 gen(18);
    uniform_real_distribution<> random_prob(0.0, 1.0);

    for (int iter = 1; ; ++iter) {
        if (no_improve > max_no_improve) break;
        
        auto current_time = chrono::steady_clock::now();
        double elapsed = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
        if (elapsed > 55) break;
        
        vector<int> iteration_best_route;
        double iteration_best_cost = 2e18;

        for (int k = 0; k < m_ants; ++k) {
            vector<int> route;
            vector<bool> unvisited(n + 1, true);
            unvisited[0] = false;
            int current_node = 0;
            double current_time = 0;

            for (int step = 0; step < n; ++step) {
                vector<int> N_i;
                vector<double> G_val, H_val;
                double sum_G = 0, sum_H = 0;
                int count_G = 0, count_H = 0;

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

                double mu = (count_G > 0) ? (sum_G / count_G) : 0;
                double nu = (count_H > 0) ? (sum_H / count_H) : 0;

                vector<double> prob(N_i.size(), 0.0);
                double prob_sum = 0;
                int best_j = -1;
                double max_eval = -1;

                // 3. Ép kiến đi tiếp dù count_G == 0
                for (size_t idx = 0; idx < N_i.size(); ++idx) {
                    int j = N_i[idx];
                    
                    // Fallback: nếu chết hết, trả g_ij = 1.0 để dùng Pheromone thuần túy dẫn đường
                    double g_ij = (count_G == 0) ? 1.0 : 0.0; 
                    double h_ij = 1.0;

                    if (count_G > 0 && G_val[idx] >= 0) {
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

                int next_node = -1;
                double q = random_prob(gen);
                
                if (q <= q0) {
                    next_node = best_j; 
                } else {
                    if (prob_sum > 0) {
                        double rand_val = random_prob(gen) * prob_sum;
                        double cumulative = 0;
                        for (size_t idx = 0; idx < N_i.size(); ++idx) {
                            cumulative += prob[idx];
                            if (cumulative >= rand_val) {
                                next_node = N_i[idx];
                                break;
                            }
                        }
                    }
                    if (next_node == -1) next_node = N_i.back();
                }

                route.push_back(next_node);
                unvisited[next_node] = false;
                
                double s_time = (current_node == 0) ? 0 : d[current_node-1];
                current_time = max(current_time + s_time + t[current_node][next_node], e[next_node-1]);
                
                tau[current_node][next_node] = (1.0 - omega) * tau[current_node][next_node] + omega * tau_0;
                current_node = next_node;
            }

            // 4. Đánh giá nghiệm phạt để rải Pheromone
            double penalized_cost = cal_cost(route, penalty);
            if (penalized_cost < iteration_best_cost) {
                iteration_best_cost = penalized_cost;
                iteration_best_route = route;
            }

            // 5. Đánh giá nghiệm khả thi tuyệt đối
            if (is_feasible(route)) {
                double true_cost = cal_cost(route, 0.0);
                if (true_cost < true_best_feasible_cost) {
                    true_best_feasible_cost = true_cost;
                }
            }
        } // Kết thúc 1 vòng lặp của m kiến

        if (iteration_best_cost < global_best_cost) {
            global_best_cost = iteration_best_cost;
            global_best_route = iteration_best_route;
            no_improve = 0;
        }
        else ++no_improve;

        // Rải Pheromone TOÀN CỤC dựa trên lộ trình tốt nhất (dù nó có trễ giờ đi nữa)
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

    // 6. Trả về nghiệm tốt nhất Khả thi nếu có, không thì trả về nghiệm Phạt tốt nhất
    return (true_best_feasible_cost != 2e18) ? true_best_feasible_cost : global_best_cost;
}

int main() {
    cin >> n;
    init();
    for (int i = 0; i < n; ++i) cin >> e[i] >> l[i] >> d[i];
    for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j) cin >> t[i][j];

    cout << solve() << endl;
}
