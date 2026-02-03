import numpy as np
from ising_core import IsingSimulation
import utils
import time
import os

def main():
    # --- 1. 硬件与实验准备 ---
    # 针对 R9000P 优化的 5 组线性规模
    N_list = [16, 32, 48, 64, 80]  # 选择的格点规模

    # 温度区间（T）集中在临界点附近
    T_range = np.array([1.9, 2.1, 2.2, 2.22, 2.24, 2.26, 2.28, 2.30, 2.4, 2.5, 2.7])
    
    # 采样设置：每个T跑5000步
    steps = 5000  # 采样步数
    burn_in = 2000  # 预热步数：确保系统跨越弛豫时间达到平衡

    # 开始计时
    start_time = time.time()

    print("=" * 50)
    print(f"格点规模: {N_list}")
    print(f"温度点数: {len(T_range)}")
    print(f"每个T的总步数: {steps + burn_in} MCS")
    print("=" * 50)

    # --- 2. 执行核心计算 ---
    # 创建保存数据的文件夹
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)

    for N in N_list:
        print(f"\n正在进行 N = {N} 的模拟...")

        # 存储当前 N 的所有结果
        all_m_avg = []
        all_e_avg = []

        # 循环遍历每个温度
        for T in T_range:
            print(f"  运行 T = {T}...")

            # 创建一个 Ising 模拟实例
            sim = IsingSimulation(L=N, T=T)

            # 运行模拟并获得平均磁化强度和能量
            m_avg, e_avg = sim.run(steps=steps, burn_in=burn_in)

            # 存储每个 T 的结果
            all_m_avg.append(m_avg)
            all_e_avg.append(e_avg)

            # 保存每个 T 的数据
            np.savez(
                os.path.join(save_dir, f"data_L{N}_T{T}.npz"),
                m_samples=sim.m_samples,
                e_samples=sim.e_samples,
                m_avg=m_avg,
                e_avg=e_avg
            )
            print(f"  数据保存完毕: data_L{N}_T{T}.npz")

        # 存储该 N 所有 T 的平均值（m_avg 和 e_avg）
        np.savez(
            os.path.join(save_dir, f"avg_L{N}.npz"),
            T=T_range,
            m_avg=np.array(all_m_avg),
            e_avg=np.array(all_e_avg)
        )
        print(f"  所有 T 的平均数据已保存: avg_L{N}.npz")

    # --- 3. 数据处理与可视化 ---
    try:
        # 调用 FSS 分析函数，生成 M-T、Cv-T、Chi-T 等图
        utils.plot_fss_analysis(
            IsingSimulation, 
            N_list=N_list, 
            T_range=T_range, 
            steps=steps, 
            burn_in=burn_in
        )
    except Exception as e:
        print(f"\n[!] 模拟运行中出现意外中断: {e}")
    else:
        end_time = time.time()
        duration = (end_time - start_time) / 3600
        print("\n" + "=" * 50)
        print(f"所有实验圆满完成！总耗时: {duration:.2f} 小时")
        print(f"请在 'data' 文件夹查看数据文件，并在 'plots' 文件夹查看 FSS 分析结果图")
        print("=" * 50)

if __name__ == "__main__":
    main()
