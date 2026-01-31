import numpy as np
import matplotlib.pyplot as plt
import time
import os
from numba import njit

# ==========================================
# ⚙️ 核心物理核：支持 M 与 G 两种动力学
# ==========================================
@njit
def dynamics_kernel(L, beta, steps, algo_type):
    """
    algo_type: 0 为 Metropolis, 1 为 Glauber
    """
    # 1. 查找表预计算
    # dE 可能取值: -8, -4, 0, 4, 8
    # 对应索引: 0, 1, 2, 3, 4
    prob_table = np.zeros(5)
    de_values = np.array([-8, -4, 0, 4, 8])
    
    for idx in range(5):
        dE = de_values[idx]
        if algo_type == 0:  # Metropolis: min(1, exp(-beta*dE))
            prob_table[idx] = min(1.0, np.exp(-beta * dE))
        else:               # Glauber: 1 / (1 + exp(beta*dE))
            prob_table[idx] = 1.0 / (1.0 + np.exp(beta * dE))

    # 2. 初始化：为了观察收敛，统一从全朝上(M=1)开始
    config = np.ones((L, L), dtype=np.int8)
    M = float(L * L)
    m_history = np.zeros(steps)

    # 3. 演化循环
    for s_idx in range(steps):
        for _ in range(L * L):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            s = config[i, j]
            nb = config[(i+1)%L, j] + config[(i-1)%L, j] + \
                 config[i, (j+1)%L] + config[i, (j-1)%L]
            dE = 2 * s * nb
            
            # 查表索引映射: dE // 4 + 2 (将 -8..8 映射到 0..4)
            p_acc = prob_table[dE // 4 + 2]
            
            if np.random.rand() < p_acc:
                config[i, j] *= -1
                M += -2 * s
        
        # 记录归一化磁化强度 |m|
        m_history[s_idx] = abs(M) / (L * L)
        
    return m_history

# ==========================================
# 🚀 自动化调度与数据保存
# ==========================================
def run_comparison():
    # 实验参数
    L = 64
    T = 2.27
    beta = 1.0 / T
    runs = 200
    steps = 30000  # Tc 附近收敛慢，步数需稍长
    
    save_dir = "data_dynamics_T2.27"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    results = {"Metropolis": [], "Glauber": []}
    
    print(f"🌟 启动动力学对比实验 | T={T}, L={L}")
    print(f"📂 存储目录: {save_dir}")

    # 分支 1: Glauber 动力学
    print("\n--- 正在运行 Glauber 动力学 (20 runs) ---")
    for r in range(runs):
        m_path = dynamics_kernel(L, beta, steps, algo_type=1)
        results["Glauber"].append(m_path)
        # 实时保存单次运行数据
        np.savez(f"{save_dir}/Glauber_run{r:02d}.npz", T=T, m_history=m_path)
        if (r+1) % 5 == 0: print(f"进度: {r+1}/{runs}")

    # 分支 2: Metropolis 算法
    print("\n--- 正在运行 Metropolis 算法 (20 runs) ---")
    for r in range(runs):
        m_path = dynamics_kernel(L, beta, steps, algo_type=0)
        results["Metropolis"].append(m_path)
        # 实时保存单次运行数据
        np.savez(f"{save_dir}/Metropolis_run{r:02d}.npz", T=T, m_history=m_path)
        if (r+1) % 5 == 0: print(f"进度: {r+1}/{runs}")

    # --- 简图输出 (判断收敛用) ---
    plt.figure(figsize=(8, 5))
    
    # 计算系综平均
    m_avg_g = np.mean(np.array(results["Glauber"]), axis=0)
    m_avg_m = np.mean(np.array(results["Metropolis"]), axis=0)
    
    plt.plot(m_avg_g, label='Glauber', color='red', alpha=0.8)
    plt.plot(m_avg_m, label='Metropolis', color='blue', alpha=0.8)
    
    plt.axhline(y=0.15, color='gray', linestyle='--', label='Expected Baseline') # 提示 0.15 的位置
    plt.xlabel('Time (MCS)')
    plt.ylabel('<|m|>')
    plt.title(f'Dynamics Comparison at T={T} (Ensemble Average n=20)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # --- 关键：保存图片 ---
    plot_path = f"{save_dir}/comparison_T2.27_L64.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight') 
    print(f"📊 简图已保存至: {plot_path}")
    
    plt.show()
    print(f"\n✅ 任务完成。数据已存入 {save_dir}。")

if __name__ == "__main__":
    run_comparison()