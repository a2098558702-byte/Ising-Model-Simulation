import numpy as np
import time
import os
from ising_core import IsingSimulation

def run_metropolis_dynamics_safe():
    # --- 1. 参数设置 ---
    N = 64
    T = 2.4
    steps = 3000
    runs = 200  # 保持与 Glauber 一致，确保对比图美观
    beta = 1.0 / T
    
    # 【要求3】新建专用文件夹
    save_dir = "metropolis_data_dynamics"
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print(f"🚀 启动 Metropolis 动力学对比测量")
    print(f"📂 保存目录: ./{save_dir}/")
    print(f"⚡ 算法: Metropolis (LUT加速版)")
    print("="*60)

    # 预计算 Metropolis 概率表 (dE > 0 时才查表)
    # dE 只有 4 和 8 两种情况需要算 exp，其他 dE<=0 概率为 1
    metro_lut = {
        4: np.exp(-beta * 4),
        8: np.exp(-beta * 8)
    }

    # 定义 Metropolis 加速步函数
    def fast_metropolis_step(sim):
        config = sim.config
        L = sim.L
        # 批量生成随机数
        rand_is = np.random.randint(0, L, L*L)
        rand_js = np.random.randint(0, L, L*L)
        rand_probs = np.random.rand(L*L)
        
        for k in range(L*L):
            i, j = rand_is[k], rand_js[k]
            s = config[i, j]
            nb_sum = (config[(i+1)%L, j] + config[(i-1)%L, j] +
                      config[i, (j+1)%L] + config[i, (j-1)%L])
            dE = 2 * s * nb_sum
            
            # Metropolis 判据
            accept = False
            if dE <= 0:
                accept = True
            elif rand_probs[k] < metro_lut[int(dE)]: # 查表
                accept = True
                
            if accept:
                config[i, j] *= -1
                sim.magnetization += -2 * s
                sim.energy += dE

    start_time = time.time()

    # --- 2. 循环 runs ---
    for r in range(runs):
        # 初始化：必须与 Glauber 一样从全序 (M=1) 开始
        sim = IsingSimulation(L=N, T=T)
        sim.config = np.ones((N, N), dtype=int)
        sim.magnetization = np.sum(sim.config)
        
        # 记录单次轨迹
        history = np.zeros(steps)
        
        for t in range(steps):
            fast_metropolis_step(sim)
            history[t] = abs(sim.magnetization_density)
            
        # 【要求2 & 4】单独保存，独立命名
        filename = f"run_{r+1:03d}.npz"
        np.savez(os.path.join(save_dir, filename), magnetization=history, t=np.arange(steps))

        # 【要求5】每 5 次输出进度
        if (r + 1) % 5 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"✅ Metropolis Run {r+1:3d}/{runs} 完成 | 耗时: {elapsed:.2f} min")

    print(f"\n✨ Metropolis 全部数据采集完毕！总耗时: {(time.time() - start_time)/60:.2f} min")

if __name__ == "__main__":
    run_metropolis_dynamics_safe()