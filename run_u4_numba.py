import numpy as np
import time
import os
from numba import njit

# --- 1. 这是被 JIT 编译的计算核心，速度直接起飞 ---
@njit
def fast_u4_kernel(L, beta, burn_in, steps):
    # 预计算 Metropolis 概率表（Numba 下使用数组索引比字典快得多）
    # 映射关系：dE=4 -> index 1, dE=8 -> index 2, dE<=0 -> index 0
    prob_table = np.array([1.0, np.exp(-4 * beta), np.exp(-8 * beta)])
    
    # 初始化
    config = np.random.choice(np.array([-1, 1]), (L, L))
    M = np.sum(config)
    m2_sum = 0.0
    m4_sum = 0.0
    
    # 总演化循环
    total_steps = burn_in + steps
    for s_idx in range(total_steps):
        for _ in range(L * L):
            # 随机选取格点
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            
            s = config[i, j]
            # 周期性边界条件
            nb = config[(i+1)%L, j] + config[(i-1)%L, j] + \
                 config[i, (j+1)%L] + config[i, (j-1)%L]
            dE = 2 * s * nb
            
            # 翻转判断
            if dE <= 0:
                config[i, j] *= -1
                M += -2 * s
            else:
                # dE 为 4 或 8，对应 prob_table 索引 1 或 2
                if np.random.rand() < prob_table[dE // 4]:
                    config[i, j] *= -1
                    M += -2 * s
        
        # 过了预热期进行采样
        if s_idx >= burn_in:
            m_abs = abs(M)
            m2_sum += m_abs**2
            m4_sum += m_abs**4
            
    return m2_sum / steps, m4_sum / steps

# --- 2. 7 小时“逆天”任务调度器 ---
def run_ultimate_overnight():
    # 既然有加速，我们直接挑战大尺寸
    L_list = [16, 32, 48, 64, 80, 128] 
    burn_in = 100000  # 增加预热确保平衡
    steps = 500000    # 50 万步极高采样，确保 U4 曲线丝滑
    
    # 极高精度的温度扫描 (40 个点，彻底消灭锯齿)
    T_range = np.linspace(2.22, 2.32, 40)
    
    save_dir = "data_ultimate_u4"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 启动 7 小时极限模拟模式 (JIT Accelerated)")
    print(f"📊 采样步数: {steps} | 温度分辨率: 40 points")
    
    total_start = time.time()

    for L in L_list:
        print(f"\n>>> 正在攻克 L={L} (规模 {L}x{L}) ...")
        u4_results = []
        start_L = time.time()
        
        for T in T_range:
            beta = 1.0 / T
            # 调用加速核
            m2_avg, m4_avg = fast_u4_kernel(L, beta, burn_in, steps)
            u4 = 1.0 - m4_avg / (3.0 * m2_avg**2)
            u4_results.append(u4)
            print(f"  [T={T:.4f}] U4 = {u4:.6f}")
            
        # 每一层 L 跑完即存盘，防止意外
        np.savez(f"{save_dir}/u4_L{L}.npz", T=T_range, u4=u4_results)
        elapsed = (time.time() - start_L) / 60
        print(f"✅ L={L} 完成！耗时: {elapsed:.2f} min")

    print(f"\n🎉 任务圆满完成！总耗时: {(time.time() - total_start) / 60:.2f} min")

if __name__ == "__main__":
    run_ultimate_overnight()