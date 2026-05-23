import os
import subprocess
import time
import resource
import matplotlib.pyplot as plt
import numpy as np


EXECUTABLES = {
    'MIP': 'linear programming/branch_and_cut.py',
    'CP': 'constrast programming/guided_local_search.py',
    'LS': 'local search/local_search',
    'TS': 'tabu search/tabu_search',
    'SA': 'stimulated annealing/stimulated_annealing',
    'ACS': 'ant colony system/acs'
}

DATA_DIR = "data/AFG_converted_travel_time_objective"
TIME_LIMIT_SEC = 60
MEMORY_LIMIT_MB = 8 * 1024
NUM_RUNS = 1

def set_process_limits():
    mem_limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (int(TIME_LIMIT_SEC) + 1, int(TIME_LIMIT_SEC) + 1))

def run_testcase(test_file, executable_path):
    test_path = os.path.join(DATA_DIR, test_file)
    
    with open(test_path, 'r') as f:
        lines = f.read().strip().split('\n')

    optimal_value = float(lines[0].strip())
    input_data = '\n'.join(lines[1:]) + '\n'

    start_time = time.perf_counter()
    status = ""
    heuristic_value = None
    gap = None
    
    try:
        command = ["python3", executable_path] if executable_path.endswith('.py') else [executable_path]
        process = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_SEC,
            preexec_fn=set_process_limits
        )
        
        elapsed_time = time.perf_counter() - start_time
        
        if process.returncode == 0:
            out_str = process.stdout.strip()
            heuristic_value = float(out_str)
            
            if heuristic_value == -1.0:
                status = "❌ No Solution"
                gap = None 
            else:
                status = "✅ AC"
                gap = (heuristic_value - optimal_value) / optimal_value * 100 if optimal_value > 0 else 0

                if abs(gap) < 1e-9:
                    gap = 0
                    
        else:
            status = f"🔴 RE/MLE"
            
    except subprocess.TimeoutExpired:
        elapsed_time = TIME_LIMIT_SEC
        status = "🟡 TLE"
    except Exception as e:
        elapsed_time = 0
        status = f"❌ Lỗi: {e}"

    return status, elapsed_time, gap

def evaluate(executable_name, executable_path):
    if not os.path.exists(executable_path):
        print(f"LỖI: Không tìm thấy file chạy '{executable_path}' của {executable_name}.")
        return 0, 0, 0

    test_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
    total_count = len(test_files)
    
    passed_count = 0
    total_time = 0
    gap_count = 0
    total_gap = 0

    for test_file in test_files:
        status, elapsed, gap = run_testcase(test_file, executable_path)  
        total_time += elapsed
        if "AC" in status:
            passed_count += 1
        if gap is not None:
            total_gap += gap
            gap_count += 1

    passed_percentage = (passed_count / total_count * 100) if total_count > 0 else 0
    avg_gap = (total_gap / gap_count) if gap_count > 0 else 0

    return passed_percentage, total_time, avg_gap

def plot_comparisons(results_data):
    algorithms = list(results_data.keys())
    pass_rates = [results_data[algo]['passed'] for algo in algorithms]
    times = [results_data[algo]['time'] for algo in algorithms]
    gaps = [results_data[algo]['gap'] for algo in algorithms]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(algorithms)))

    dataset_name = os.path.basename(os.path.normpath(DATA_DIR))
    save_dir = os.path.join('evaluation', 'results', dataset_name)

    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # ĐỒ THỊ 1: TỶ LỆ VƯỢT QUA TESTCASE
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(algorithms, pass_rates, color=colors)
    plt.title('Tỷ lệ vượt qua Testcase (%)', fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.ylabel('Phần trăm (%)')
    plt.bar_label(bars1, fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'chart_pass_rate.png'), dpi=300) 

    # ---------------------------------------------------------
    # ĐỒ THỊ 2: TỔNG THỜI GIAN CHẠY
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bars2 = plt.bar(algorithms, times, color=colors)
    plt.title('Tổng thời gian chạy (Giây)', fontsize=14, fontweight='bold')
    plt.ylabel('Giây (s)')
    plt.bar_label(bars2, fmt='%.4fs', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'chart_time.png'), dpi=300)

    # ---------------------------------------------------------
    # ĐỒ THỊ 3: ĐỘ LỆCH TRUNG BÌNH (GAP)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bars3 = plt.bar(algorithms, gaps, color=colors)
    plt.title('Độ lệch trung bình (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Độ lệch (%)')
    plt.bar_label(bars3, fmt='%.4f%%', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'chart_gap.png'), dpi=300)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"LỖI: Thư mục '{DATA_DIR}' không tồn tại hoặc đang trống.")
        exit()

    final_results = {}

    for algo_name, algo_path in EXECUTABLES.items():
        print(f"==================================================")
        print(f"Đang đánh giá thuật toán: {algo_name}")
        
        passed_history = []
        time_history = []
        gap_history = []

        for i in range(NUM_RUNS):
            passed, total_time, avg_gap = evaluate(algo_name, algo_path)
            passed_history.append(passed)
            time_history.append(total_time)
            gap_history.append(avg_gap)
        
        avg_passed = sum(passed_history) / len(passed_history)
        avg_total_time = sum(time_history) / len(time_history)
        avg_total_gap = sum(gap_history) / len(gap_history)

        final_results[algo_name] = {
            'passed': avg_passed,
            'time': avg_total_time,
            'gap': avg_total_gap
        }

        print(f"[KẾT QUẢ {algo_name}]")
        print(f"Tỷ lệ qua test: {avg_passed:.1f}% | Thời gian tổng: {avg_total_time:.4f}s | Gap trung bình: {avg_total_gap:.4f}%\n")

    if final_results:
        print("Đang vẽ biểu đồ so sánh...")
        plot_comparisons(final_results)
