import os
import subprocess
import time
import resource


EXECUTABLE = "linear programming/guided-local-search.py"
DATA_DIR = "data" 
TIME_LIMIT_SEC = 60
MEMORY_LIMIT_MB = 1024

def set_process_limits():
    mem_limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (int(TIME_LIMIT_SEC) + 1, int(TIME_LIMIT_SEC) + 1))

def run_testcase(test_file):
    test_path = os.path.join(DATA_DIR, test_file)
    
    with open(test_path, 'r') as f:
        lines = f.read().strip().split('\n')

    optimal_value = float(lines[0].strip())
    input_data = '\n'.join(lines[1:]) + '\n'

    start_time = time.perf_counter()
    status = ""
    output = ""
    
    try:
        command = ["python3", EXECUTABLE] if EXECUTABLE.endswith('.py') else [EXECUTABLE]
        process = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_SEC,
            # preexec_fn=set_process_limits
        )
        
        elapsed_time = time.perf_counter() - start_time
        
        if process.returncode == 0:
            status = "✅ AC"
            output = process.stdout.strip()
        else:
            status = f"🔴 RE/MLE"
            
    except subprocess.TimeoutExpired:
        elapsed_time = TIME_LIMIT_SEC
        status = "🟡 TLE"
    except Exception as e:
        elapsed_time = 0
        status = f"❌ Lỗi Hệ thống: {e}"

    return status, elapsed_time, output

def main():
    if not os.path.exists(EXECUTABLE):
        print(f"LỖI: Không tìm thấy file chạy '{EXECUTABLE}'. Vui lòng biên dịch code C++ trước (g++ dp.cpp -o dp.out).")
        return

    if not os.path.exists(DATA_DIR):
        print(f"LỖI: Không tìm thấy thư mục '{DATA_DIR}'.")
        return

    test_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
    
    if not test_files:
        print(f"CẢNH BÁO: Thư mục '{DATA_DIR}' đang trống.")
        return

    print("=" * 60)
    
    passed_count = 0
    total_count = len(test_files)
    total_time = 0
    
    print(f"{'Testcase':<20} | {'Độ lệch':<30} | {'Thời gian (s)'}")
    print("-" * 60)

    for test_file in test_files:
        status, elapsed, output, optimal_value = run_testcase(test_file)
        
        name = os.path.splitext(test_file)[0]

        display_str = status
        if "AC" in status:
            passed_count += 1
            output_val = float(output)
            deviation = (optimal_value - output_val) / optimal_value if optimal_value != 0 else (optimal_value - output_val)
            display_str = f"{deviation:.6f}"
        
        print(f"{name:<20} | {display_str:<30} | {elapsed:.4f}s")

        total_time += elapsed
        
        if "AC" in status:
            passed_count += 1

    print("=" * 60)
    print(f"Kết quả: {passed_count} / {total_count} testcases ({(passed_count/total_count)*100:.1f}%) | Tổng thời gian: {total_time:.4f}s")

if __name__ == "__main__":
    main()
