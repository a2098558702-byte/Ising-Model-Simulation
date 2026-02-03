# run_fss_overnight.py
import numpy as np
from ising_core import IsingSimulation
import utils
import time

def main():
    # --- 1. 硬件与实验准备 ---
    # 针对 R9000P 优化的 7 组线性规模，确保覆盖标度区间
    N_list = [16, 24, 32, 40, 48, 56, 64]
    
    # 温度区间集中在 Onsager 临界点 (2.269) 附近
    T_range = np.linspace(1.8, 3.2, 35)
    
    # 兼顾精度与时间的采样设置
    steps = 6000       # 采样步数：确保统计平均值收敛
    burn_in = 3000     # 预热步数：确保系统跨越弛豫时间达到平衡
    
    start_time = time.time()
    
    print("="*50)
    print(f"格点规模: {N_list}")
    print(f"温度点数: {len(T_range)}")
    print(f"每点总计: {steps + burn_in} MCS")
    print("="*50)

    # --- 2. 执行核心计算 ---
    # 调用我们讨论过的 FSS 分析接口，计算 M, Cv, Chi
    try:
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
        print("\n" + "="*50)
        print(f"所有实验圆满完成！总耗时: {duration:.2f} 小时")
        print("请在 'plots' 文件夹查看 FSS_Overnight_Result.png")
        print("="*50)

if __name__ == "__main__":
    main()