# run.py
import numpy as np
from ising_core import IsingSimulation
import utils

def main():
    # --- 1. 参数全局设置 ---
    N = 30              # 格点规模 30x30
    steps = 2000        # 采样步数
    burn_in = 800       # 预热步数
    
    # --- 2. 实验一：深入观察临界点附近的动力学 ---
    T_critical = 2.27
    # 注意：这里的 \n 不能加 r，否则不会换行
    print(f"\n>>> 阶段 1：观察临界点 T={T_critical} 的演化 (从无规开始)...")
    
    sim_evolve = IsingSimulation(N=N)
    # 【关键】：burn_in=0 以观察从无序到有序的完整过程
    sim_evolve.run(T=T_critical, steps=steps, burn_in=0)
    
    utils.plot_magnetization_history(sim_evolve, T=T_critical)
    utils.plot_energy_history(sim_evolve, T=T_critical)
    utils.plot_config(sim_evolve, T=T_critical)
    utils.save_simulation_gif(sim_evolve, T=T_critical, frames=80)

    # --- 3. 实验二：全温度扫描 (稳态) ---
    print(f"\n>>> 阶段 2：全自动扫描稳态相变曲线 (N={N})...")
    T_range = np.linspace(1.0, 4.0, 10)
    
    # 稳态扫描需设置足够的 burn_in 以确保达到平衡态
    utils.plot_phase_transition(IsingSimulation, N=N, T_range=T_range, 
                                steps=steps, burn_in=burn_in)

    print("\n[所有实验圆满完成！]")
    print("请在 'plots' 文件夹下查看生成的图片和动图。")

if __name__ == "__main__":
    main()