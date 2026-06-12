import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_comparisons(json_file):
    if not os.path.exists(json_file):
        print(f"LỖI: Không tìm thấy file '{json_file}'")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("File JSON trống, không có dữ liệu để vẽ.")
        return

    dataset_name = os.path.dirname(json_file)
    save_dir = os.path.join(dataset_name,'charts')
    os.makedirs(save_dir, exist_ok=True)

    algorithms = list(data.keys())
    pass_rates = [data[algo]['passed'] for algo in algorithms]
    times = [data[algo]['time'] for algo in algorithms]
    gaps = [data[algo]['gap'] for algo in algorithms]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(algorithms)))

    # --- ĐỒ THỊ 1: TỶ LỆ VƯỢT QUA TESTCASE ---
    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(algorithms, pass_rates, color=colors)
    plt.title(f'Tỷ lệ vượt qua Testcase (%)', fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.ylabel('Phần trăm (%)')
    plt.bar_label(bars1, fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'chart_pass_rate.png'), dpi=300) 
    plt.close()

    # --- ĐỒ THỊ 2: THỜI GIAN CHẠY ---
    plt.figure(figsize=(8, 6))
    bars2 = plt.bar(algorithms, times, color=colors)
    plt.title(f'Tổng thời gian chạy (Giây)', fontsize=14, fontweight='bold')
    plt.ylabel('Giây (s)')
    plt.bar_label(bars2, fmt='%.4fs', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'chart_time.png'), dpi=300)
    plt.close()

    # --- ĐỒ THỊ 3: ĐỘ LỆCH TRUNG BÌNH ---
    plt.figure(figsize=(8, 6))
    bars3 = plt.bar(algorithms, gaps, color=colors)
    plt.title(f'Độ lệch trung bình (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Độ lệch (%)')
    plt.bar_label(bars3, fmt='%.4f%%', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'chart_gap.png'), dpi=300)
    plt.close()

    print(f"Đã lưu 3 biểu đồ tại: {save_dir}")

if __name__ == "__main__":
    json_file = 'evaluation/results/Dumas/travel_time/result.json'
    plot_comparisons(json_file)
