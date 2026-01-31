import numpy as np 
import time
import os
from numba import njit
from datetime import datetime

# --- 1. 升级版 JIT 核心：全物理量追踪 ---
@njit   # 装饰器
def fast_critical_kernel(L, beta, burn_in, steps):
    # 预计算 Metropolis 概率表，进行查表法优化
    prob_table = np.array([1.0, np.exp(-4 * beta), np.exp(-8 * beta)])
    # np.exp()太耗时
    # 必须用np.array()把列表转换为NumPy数组，因为njit不支持Python List
    # np.array([1, 2, 3]) 里面的列表只能放同类，读取快很多
    # 1. 初始化晶格
    config = np.random.choice(np.array([-1, 1]), (L, L), p=[0.5, 0.5])
    # np.array([-1, 1]) 构建候选池; (L, L)定义形状, p控制概率, 0.5时可以不写

    M = np.sum(config)   # 
    
    # 2. 计算初始总能量 E (为了后续算 Cv)
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = config[i, j]
            # 为了防止重复计算，只算右边和下边的邻居
            nb = config[i, (j+1)%L] + config[(i+1)%L, j]
            E += -s * nb
    
    # 累加器初始化
    m_abs_sum = 0.0    # 用于 <|M|>
    m_sq_sum = 0.0     # 用于 <M^2> -> 磁化率
    m_quad_sum = 0.0   # 用于 <M^4> -> U4
    e_sum = 0.0        # 用于 <E>
    e_sq_sum = 0.0     # 用于 <E^2> -> 比热 Cv
    
    total_steps = burn_in + steps
    
    # 3. 演化循环
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
            
            # Metropolis 判据
            accept = False
            if dE <= 0:
                accept = True
            else:
                if np.random.rand() < prob_table[dE // 4]:
                    accept = True
            
            # 如果接受翻转
            if accept:
                config[i, j] *= -1
                M += -2 * s
                E += dE  # 关键：实时更新能量，不需要每次重算
        
        # 4. 采样记录 (过预热期后)
        if s_idx >= burn_in:
            m_abs = abs(M)
            e_val = E
            
            m_abs_sum += m_abs
            m_sq_sum += m_abs**2
            m_quad_sum += m_abs**4
            e_sum += e_val
            e_sq_sum += e_val**2
            
    # 返回所有平均值
    return (m_abs_sum / steps, 
            m_sq_sum / steps, 
            m_quad_sum / steps, 
            e_sum / steps, 
            e_sq_sum / steps)

# --- 2. 定点高精任务调度器 ---
def run_critical_exponents():
    # 参数设置
    L_list = [16, 32, 48, 64, 80, 128] 
    burn_in = 100000        # 10万步预热
    steps = 1000000         # 100万步采样 (精度极高)
    T_c = 2.2685            # 锁死临界温度
    beta = 1.0 / T_c
    
    # 自动生成唯一文件夹名 (防止覆盖)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"data_critical_fit_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 启动临界指数提取模式 (Fixed T={T_c})")
    print(f"📂 数据将保存至: {save_dir}")
    print(f"📊 采样步数: {steps} (100万步) | 预热: {burn_in}")
    
    total_start = time.time()

    for L in L_list:
        print(f"\n>>> [L={L}] 正在进行高精模拟...")
        start_L = time.time()
        
        # 运行 Numba 核
        m_abs, m_sq, m_quad, e_avg, e_sq = fast_critical_kernel(L, beta, burn_in, steps)
        
        # 计算导出量 (仅供屏幕显示，原始数据全部保存)
        u4 = 1.0 - m_quad / (3.0 * m_sq**2)
        chi = (m_sq - m_abs**2) * beta * (L**2) # 简略估算，后续处理用严谨公式
        cv = (e_sq - e_avg**2) * (beta**2) / (L**2)
        
        print(f"   <|M|> : {m_abs:.4f}")
        print(f"   Chi   : {chi:.2f}")
        print(f"   Cv    : {cv:.4f}")
        print(f"   U4    : {u4:.5f}")
        
        # 保存所有原始矩，方便后续做拟合
        # 变量名与你的需求一一对应
        np.savez(f"{save_dir}/fit_data_L{L}.npz", 
                 T=T_c,
                 L=L,
                 m_abs_mean=m_abs,   # <|M|> 用于计算 beta/nu
                 m_sq_mean=m_sq,     # <M^2> 用于计算 Chi ~ gamma/nu
                 m_quad_mean=m_quad, # <M^4> 用于检查 U4
                 e_mean=e_avg,       # <E>
                 e_sq_mean=e_sq      # <E^2> 用于计算 Cv ~ alpha/nu
                 )
        
        elapsed = (time.time() - start_L)
        print(f"✅ L={L} 完成 (耗时 {elapsed:.2f}s)")

    print(f"\n🎉 所有数据已保存至 {save_dir}，总耗时: {(time.time() - total_start)/60:.2f} min")

if __name__ == "__main__":
    run_critical_exponents()