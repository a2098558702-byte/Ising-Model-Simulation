import numpy as np
import matplotlib.pyplot as plt
from ising_core import IsingSimulation
import time
import os

def run_glauber_safe_and_fast():
    # --- 1. 参数设置 ---
    N = 64
    T = 2.4
    steps = 3000   # 步数
    runs = 200     # 总次数
    beta = 1.0 / T
    
    # 【重点检查 1】新建一个完全不同的文件夹，避免混淆
    save_dir = "data_dynamics_safe"
    # 【重点检查 2】自动创建文件夹
    os.makedirs(save_dir, exist_ok=True)

    print("="*60)
    print(f"🛡️  启动【独立文件流】安全模拟")
    print(f"📂  数据文件夹: ./{save_dir}/ (确保与旧数据隔离)")
    print(f"💾  保存机制: 每跑完一次，生成一个独立文件 (如 run_001.npz)")
    print("="*60)

    # 预计算 Glauber 概率表 (加速引擎)
    glauber_lut = {dE: 1.0 / (1.0 + np.exp(beta * dE)) for dE in [-8, -4, 0, 4, 8]}

    # 结果容器：用于最后画总图，但中间数据会独立保存
    all_runs_history = np.zeros((runs, steps))

    # 定义查表加速函数 (注入式优化)
    def fast_glauber_step(sim):
        config = sim.config
        L = sim.L
        # 批量生成随机数 (加速)
        rand_is = np.random.randint(0, L, L*L)
        rand_js = np.random.randint(0, L, L*L)
        rand_probs = np.random.rand(L*L)
        
        for k in range(L*L):
            i, j = rand_is[k], rand_js[k]
            s = config[i, j]
            # 计算邻居和
            nb_sum = (config[(i+1)%L, j] + config[(i-1)%L, j] +
                      config[i, (j+1)%L] + config[i, (j-1)%L])
            dE = 2.0 * s * nb_sum
            # 查表
            if rand_probs[k] < glauber_lut[int(dE)]:
                config[i, j] *= -1
                sim.magnetization += -2 * s
                sim.energy += dE

    start_time = time.time()

    # --- 2. 循环实验 ---
    for r in range(runs):
        # 初始化
        sim = IsingSimulation(L=N, T=T)
        sim.config = np.ones((N, N), dtype=int)
        sim.magnetization = np.sum(sim.config)
        sim.energy = sim._compute_total_energy()

        # 单次轨迹记录器
        current_history = np.zeros(steps)

        # 跑模拟
        for t in range(steps):
            fast_glauber_step(sim)
            # 使用属性访问，避开 AttributeError
            current_history[t] = abs(sim.magnetization_density)
        
        # 记录到总内存以备最后画图
        all_runs_history[r, :] = current_history

        # 【重点检查 3】每一次都保存为独立文件！
        # 文件名类似: run_001.npz, run_002.npz ... 绝不重复覆盖
        filename = f"run_{r+1:03d}.npz"
        file_path = os.path.join(save_dir, filename)
        
        np.savez(file_path, magnetization=current_history, t=np.arange(steps))

        # 每 5 次发一条文字，让你安心
        if (r + 1) % 5 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"✅ [已生成文件] {filename} | 进度: {r+1}/{runs} | 耗时: {elapsed:.2f} min")

    # --- 3. 最后汇总画图 ---
    print("\n📦 所有独立文件保存完毕，正在生成汇总对比图...")
    glauber_avg = np.mean(all_runs_history, axis=0)
    
    # 保存一个总的平均值方便调用
    np.savez(os.path.join(save_dir, "avg_summary.npz"), m_avg=glauber_avg)

    plt.figure(figsize=(10, 6))
    plt.plot(glauber_avg, color='#2ca02c', linewidth=2, label=f'Glauber (Avg of {runs} files)')
    plt.title(f"Glauber Dynamics ({runs} runs, L={N}, T={T})")
    plt.xlabel("Time (MCS)")
    plt.ylabel("Magnetization |M|")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    img_path = os.path.join(save_dir, "final_plot.png")
    plt.savefig(img_path, dpi=300)
    
    print(f"✨ 任务完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟")
    print(f"📈 最终图片已保存至: {img_path}")

if __name__ == "__main__":
    run_glauber_safe_and_fast()