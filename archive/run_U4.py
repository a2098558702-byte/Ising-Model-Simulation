import numpy as np
import time
import os
from ising_core import IsingSimulation

def run_u4_crossing_smart():
    # --- 1. 高性价比参数 ---
    L_list = [16, 32, 48, 64, 80] # 加上 80 也行，大概多花 20 分钟
    burn_in = 50000
    steps = 250000  # 25万步，对于 U4 这种高阶量是“及格线”，但对于你的时间预算是“完美线”
    
    # 精简温度列表 (16个点)
    T_range = np.unique(np.concatenate([
        np.linspace(2.20, 2.25, 3),
        np.linspace(2.255, 2.285, 10), # 核心区
        np.linspace(2.29, 2.35, 3)
    ]))
    
    save_dir = "data_u4_crossing"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🎯 启动 U4 狙击模式")
    print(f"🔥 步数: {burn_in} + {steps} | 温度点: {len(T_range)} 个")
    
    total_start = time.time()

    for L in L_list:
        print(f"\n>>> 正在计算 L={L} ...")
        u4_list = []
        T_record = []
        
        start_L = time.time()
        
        for T in T_range:
            beta = 1.0 / T
            # Metropolis LUT (预计算概率)
            # 只有 dE=4, 8 需要判定概率，dE<=0 必翻
            lut = {4: np.exp(-4*beta), 8: np.exp(-8*beta)}
            
            sim = IsingSimulation(L=L, T=T)
            # 随机初始化比全序初始化在临界区收敛稍快
            sim.config = np.random.choice([-1, 1], size=(L, L))
            sim.magnetization = np.sum(sim.config)
            sim.energy = sim._compute_total_energy()
            
            # --- 极速循环 (内联优化) ---
            config = sim.config
            M = sim.magnetization
            
            # 统计量
            m2_sum = 0.0
            m4_sum = 0.0
            
            # 预热
            for _ in range(burn_in):
                for _ in range(L*L):
                    r_i, r_j = np.random.randint(0, L, 2)
                    s = config[r_i, r_j]
                    nb = config[(r_i+1)%L, r_j] + config[(r_i-1)%L, r_j] + \
                         config[r_i, (r_j+1)%L] + config[r_i, (r_j-1)%L]
                    dE = 2 * s * nb
                    if dE <= 0 or np.random.rand() < lut.get(dE, 0): # get处理dE>0但不是4/8的异常(虽然不会有)
                        config[r_i, r_j] *= -1
                        M += -2 * s
            
            # 采样
            for _ in range(steps):
                for _ in range(L*L):
                    r_i, r_j = np.random.randint(0, L, 2)
                    s = config[r_i, r_j]
                    nb = config[(r_i+1)%L, r_j] + config[(r_i-1)%L, r_j] + \
                         config[r_i, (r_j+1)%L] + config[r_i, (r_j-1)%L]
                    dE = 2 * s * nb
                    # Metropolis 判断
                    if dE <= 0:
                        config[r_i, r_j] *= -1
                        M += -2 * s
                    elif np.random.rand() < lut[int(dE)]:
                        config[r_i, r_j] *= -1
                        M += -2 * s
                
                # 采样累加 (注意：M是总磁矩)
                m_abs = abs(M)
                m2_sum += m_abs**2
                m4_sum += m_abs**4
            
            # 计算 U4
            m2_avg = m2_sum / steps
            m4_avg = m4_sum / steps
            u4 = 1.0 - m4_avg / (3.0 * m2_avg**2)
            
            u4_list.append(u4)
            T_record.append(T)
            
            # 简单进度条
            print(f"   T={T:.3f} | U4={u4:.5f}")

        # 存盘
        np.savez(f"{save_dir}/u4_L{L}.npz", T=T_record, u4=u4_list)
        print(f"✅ L={L} 完成 | 耗时 {(time.time()-start_L)/60:.1f} min")

    print(f"🎉 全部完成，总耗时 {(time.time()-total_start)/60:.1f} min")

if __name__ == "__main__":
    run_u4_crossing_smart()
