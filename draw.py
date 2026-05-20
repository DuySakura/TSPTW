import os
import numpy as np
import matplotlib.pyplot as plt


def plot_comparisons():
    algorithms = list(reversed(['MIP', 'CP', 'Nhánh cận', 'Quy hoạch động', 'Local search', 'Tabu search', 'Stimulated Annealing']))
    pass_rates = list(reversed([70, 60, 30, 20, 100, 100, 100]))
    times = list(reversed([566.15, 23.246, 2115.349, 3.977, 0.821, 2.266, 4.959]))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(algorithms)))

    os.makedirs('results', exist_ok=True)

    # ---------------------------------------------------------
    # ĐỒ THỊ 1: TỶ LỆ VƯỢT QUA TESTCASE
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bars1 = plt.barh(algorithms, pass_rates, color=colors)
    plt.title('Tỷ lệ vượt qua Testcase (%)', fontsize=14, fontweight='bold')
    plt.xlim(0, 110)
    plt.xlabel('Phần trăm (%)')
    plt.bar_label(bars1, fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig('results/chart_pass_rate.png', dpi=300, bbox_inches='tight')

    # ---------------------------------------------------------
    # ĐỒ THỊ 2: TỔNG THỜI GIAN CHẠY
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bars2 = plt.barh(algorithms, times, color=colors)
    plt.title('Tổng thời gian chạy (s)', fontsize=14, fontweight='bold')
    plt.xlim(0, max(times) * 1.2)
    plt.xlabel('Giây (s)')
    plt.bar_label(bars2, fmt='%.4fs', padding=3)
    plt.tight_layout()
    plt.savefig('results/chart_time.png', dpi=300, )

plot_comparisons()
