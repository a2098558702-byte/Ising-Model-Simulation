import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.animation as animation

def plot_magnetization_history(sim, T, save_dir='plots'):
    """
    接口 1：专门绘制磁化强度随时间演化的曲线 (M-t)
    """
    if not sim.m_samples:
        print("Error: No magnetization samples found.")
        return

    m_data = np.array(sim.m_samples)
    steps = np.arange(len(m_data))

    plt.figure(figsize=(10, 4))
    plt.plot(steps, m_data, color='#1f77b4', linewidth=1, label='Magnetization')
    plt.axhline(y=np.mean(m_data), color='orange', linestyle='--', label=f'Mean: {np.mean(m_data):.3f}')
    
    plt.title(f'Magnetization Evolution (L={sim.N}, T={T})')
    plt.xlabel('Sampling Steps')
    plt.ylabel(r'Magnetization $M$')  # 已加 r
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # 自动保存
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"M_history_T{T}.png"), dpi=300)
    plt.show()

def plot_energy_history(sim, T, save_dir='plots'):
    """
    接口 2：专门绘制能量随时间演化的曲线 (E-t)
    """
    if not sim.e_samples:
        print("Error: No energy samples found.")
        return

    e_data = np.array(sim.e_samples)
    steps = np.arange(len(e_data))

    plt.figure(figsize=(10, 4))
    plt.plot(steps, e_data, color='#d62728', linewidth=1, label='Energy')
    plt.axhline(y=np.mean(e_data), color='green', linestyle='--', label=f'Mean: {np.mean(e_data):.3f}')
    
    plt.title(f'Energy Evolution (L={sim.N}, T={T})')
    plt.xlabel('Sampling Steps')
    plt.ylabel(r'Energy $E$')  # 已加 r
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # 自动保存
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"E_history_T{T}.png"), dpi=300)
    plt.show()
    
def plot_phase_transition(sim_class, N, T_range, steps=2000, burn_in=1000, save_dir='plots'):
    """
    接口 3：全自动扫描多个温度，绘制 M-T 相变曲线
    sim_class: 传入你的类名 IsingSimulation (不需要实例化)
    N: 格点规模
    T_range: 温度列表 (例如 np.linspace(1.5, 3.5, 20))
    """
    m_averages = []
    
    print(f"开始扫描相变曲线 (N={N})...")
    
    for T in T_range:
        # 1. 针对每个温度，初始化一个新的模拟实例
        sim = sim_class(N=N)
        
        # 2. 运行模拟 (预热 + 采样)
        sim.run(T, steps=steps, burn_in=burn_in)
        
        # 3. 计算该温度下的磁化强度绝对值的平均值
        m_abs_avg = np.mean(np.abs(sim.m_samples))
        m_averages.append(m_abs_avg)
        
        print(f"  进度: T = {T:.2f} | <|M|> = {m_abs_avg:.3f}")

    # --- 绘图逻辑 ---
    plt.figure(figsize=(8, 5))
    
    # 画出模拟数据点和连接线
    plt.plot(T_range, m_averages, 'o-', color='#1f77b4', markersize=6, label='Simulation')
    
    # 标注理论上的临界温度 (Onsager 临界点)
    # 关键修正：必须加 r，防止 \approx 被转义
    plt.axvline(x=2.269, color='red', linestyle='--', alpha=0.7, label=r'Onsager $T_c \approx 2.27$') 
    
    # f-string 配合 r 使用 rf
    plt.title(rf'Phase Transition: Magnetization vs Temperature ($L={N}$)') 
    plt.xlabel('Temperature $T$')
    plt.ylabel(r'Average $|M|$') # 已加 r
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 自动保存
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "phase_transition_MT.png")
    plt.savefig(save_path, dpi=300)
    
    print(f"\n成功：相变大图已保存至 {save_path}")
    plt.show()
    
def plot_config(sim, T, save_dir='plots'):
    """
    接口 4：绘制当前的自旋构型图（黑白格点图）
    """
    plt.figure(figsize=(6, 6))
    
    plt.imshow(sim.config, cmap='Greys', interpolation='nearest')
    plt.title(f"Spin Configuration (L={sim.N}, T={T})", fontsize=13)
    plt.axis('off') 
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f"config_L{sim.N}_T{T}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"构型图已保存: {save_path}")
    plt.show()    
    
def save_simulation_gif(sim, T, frames=100, interval=50, save_dir='plots'):
    """
    接口 5：生成系统演化的动态 GIF
    """
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(sim.config, cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.title(f"Dynamic Evolution at T={T}")

    def update(frame):
        for _ in range(10):
            sim.metropolis_step(T)
        im.set_array(sim.config)
        return [im]

    print(f"正在生成 T={T} 的演化动图...")
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"evolution_T{T}.gif")
    
    ani.save(save_path, writer='pillow')
    plt.close() 
    print(f"动图已保存: {save_path}")
    
def plot_fss_analysis(sim_class, N_list, T_range, steps=5000, burn_in=2000):
    """
    接口6：进行有限尺寸标度分析
    计算不同 N 下的磁化率 Chi 和比热 Cv
    """
    results = {N: {'M': [], 'Cv': [], 'Chi': []} for N in N_list}
    
    for N in N_list:
        print(f"\n开始计算规模 N={N}...")
        for T in T_range:
            beta = 1.0 / T
            sim = sim_class(N=N)
            sim.run(T, steps=steps, burn_in=burn_in)
            
            # 磁化强度均值
            m_abs = np.mean(np.abs(sim.m_samples))
            
            # 利用涨落-耗散定理计算比热 Cv 和 磁化率 Chi
            # Cv = (beta^2 / N^2) * var(E)
            # Chi = (beta / N^2) * var(M)
            cv = (beta**2) * np.var(sim.e_samples) * (N**2) 
            chi = beta * np.var(np.abs(sim.m_samples)) * (N**2)
            
            results[N]['M'].append(m_abs)
            results[N]['Cv'].append(cv)
            results[N]['Chi'].append(chi)
            print(f"  N={N}, T={T:.2f} 已完成")

    # 绘图逻辑 (建议保存为三个子图)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for N in N_list:
        axes[0].plot(T_range, results[N]['M'], 'o-', label=f'N={N}')
        axes[1].plot(T_range, results[N]['Cv'], 's-', label=f'N={N}')
        axes[2].plot(T_range, results[N]['Chi'], '^-', label=f'N={N}')
    
    axes[0].set_title('Magnetization $|M|$')
    axes[1].set_title('Specific Heat $C_v$')
    axes[2].set_title('Magnetic Susceptibility $\chi$')
    for ax in axes: 
        ax.legend(); ax.grid(True, linestyle=':')
        ax.axvline(x=2.269, color='r', linestyle='--')
        
    plt.savefig('plots/FSS_Analysis.png', dpi=300)
    plt.show()