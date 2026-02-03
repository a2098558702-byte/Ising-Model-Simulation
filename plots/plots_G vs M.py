import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. 设置 Nature/Science 风格 (Sans-Serif)
# ==========================================
# 设置字体族为无衬线
plt.rcParams['font.family'] = 'sans-serif'
# 优先使用 Arial (Windows标准) 或 Helvetica (Mac标准)
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# 让数学公式（如 tau, M）也使用无衬线字体，保持视觉一致
# 'stixsans' 是专门配合无衬线体的数学字库
# 如果报错，可以改回 'dejavusans' (Matplotlib 默认)
plt.rcParams['mathtext.fontset'] = 'stixsans' 

# 刻度设置保持不变（这张图也是刻度朝内，四面都有）
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# 增加刻度线的宽度，让它看起来更像图里那么“硬朗”
plt.rcParams['axes.linewidth'] = 1.2 # 边框变粗
plt.rcParams['xtick.major.width'] = 1.2 # 刻度变粗
plt.rcParams['ytick.major.width'] = 1.2

def load_data(folder):
    """读取文件夹下所有 .npz 文件并计算平均值和标准差"""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npz') and 'avg' not in f])
    print(f"正在从 {folder} 读取 {len(files)} 个文件...")
    
    data_list = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            # 兼容不同的键名 key
            loaded = np.load(path)
            if 'magnetization' in loaded:
                data_list.append(loaded['magnetization'])
            elif 'history' in loaded: # 兼容旧版
                 # history可能是 [runs, steps] 或者单次
                 d = loaded['history']
                 if len(d.shape) > 1: data_list.extend(d)
                 else: data_list.append(d)
        except:
            pass
            
    data_matrix = np.array(data_list) # [Runs, Steps]
    t = np.arange(data_matrix.shape[1])
    m_avg = np.mean(data_matrix, axis=0)
    m_std = np.std(data_matrix, axis=0)
    return t, m_avg, m_std

def exponential_decay(t, tau, A, C):
    """拟合函数: M(t) = A * exp(-t/tau) + C"""
    return A * np.exp(-t / tau) + C

def analyze_and_plot():
    # 文件夹路径 (请确保这两个文件夹存在)
    dir_glauber = "data_dynamics_safe"     # 刚才跑的 Glauber
    dir_metro = "metropolis_data_dynamics" # 现在跑的 Metropolis
    
    # 1. 读取数据
    t, m_gl, std_gl = load_data(dir_glauber)
    _, m_me, std_me = load_data(dir_metro)
    
    # 2. 计算特征时间 tau (截取前段下降区，例如前300步)
    # 注意：只拟合下降最快的区间，避开后面的平台噪音
    fit_limit = 500
    
    # 拟合 Metropolis
    popt_me, _ = curve_fit(exponential_decay, t[:fit_limit], m_me[:fit_limit],
                           p0=[100, 0.8, 0.2])
    tau_me = popt_me[0]
    
    # 拟合 Glauber
    popt_gl, _ = curve_fit(exponential_decay, t[:fit_limit], m_gl[:fit_limit],
                           p0=[100, 0.8, 0.2])
    tau_gl = popt_gl[0]
    
    print("="*40)
    print(f"📊 物理结果分析 (T=2.4, L=64)")
    print(f"Metropolis 特征时间 tau ≈ {tau_me:.2f} MCS")
    print(f"Glauber    特征时间 tau ≈ {tau_gl:.2f} MCS")
    print(f"速率对比: Metropolis 比 Glauber 快 {tau_gl/tau_me:.2f} 倍")
    print("="*40)

# 3. 绘图
    # 刻度朝内，且上下左右都有
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    
       
    plt.figure(figsize=(10, 7), dpi=150) # 加上 dpi 更清晰

    # --- 关键修改 1: 创建切片掩码，只取前 1000 步的数据 ---
    mask = t
    t_plot = t[mask]
    m_me_plot = m_me[mask]
    m_gl_plot = m_gl[mask]
    
    # 假设 std 也是数组，如果 std 是常数则不需要切片
    # 为了保险，这里假设 std_me 和 std_gl 可能是标量也可能是数组
    # 如果是标量直接用，如果是数组则切片
    std_me_plot = std_me[mask] if hasattr(std_me, '__len__') and len(std_me) == len(t) else std_me
    std_gl_plot = std_gl[mask] if hasattr(std_gl, '__len__') and len(std_gl) == len(t) else std_gl
    
    # 画 Metropolis (使用切片后的数据)
    plt.plot(t_plot, m_me_plot, label=f'Metropolis ($\\tau \\approx {tau_me:.1f}$)', color='#1f77b4', linewidth=2.4)
    # 修复了原代码 m_me - m_me 的笔误，改为 m_me - std_me
    plt.fill_between(t_plot, m_me_plot - std_me_plot, m_me_plot + std_me_plot, color='#1f77b4', alpha=0.1)
    
    # 画 Glauber (使用切片后的数据)
    plt.plot(t_plot, m_gl_plot, label=f'Glauber ($\\tau \\approx {tau_gl:.1f}$)', color='#d62728', linewidth=2.4)
    plt.fill_between(t_plot, m_gl_plot - std_gl_plot, m_gl_plot + std_gl_plot, color='#d62728', alpha=0.1)
    
    # 装饰图表
    plt.xlabel('Time (MCS)', fontsize=14)
    plt.ylabel('Magnetization $|M|$', fontsize=14)
    # 一般不用标题，而是写在图的下面
    # plt.title(f'Dynamics Relaxation Comparison ($L=64, T=2.4$)\nMetropolis vs Glauber', fontsize=16)
    plt.legend(fontsize=12, loc='upper right') # 图例通常放在右上角
    plt.grid(True, alpha=0.3, linestyle='--') # 虚线网格更优雅
    
    # --- 关键修改 2: 锁定坐标轴范围，制造左右各 50 的留白 ---
    plt.xlim(0, 3000) 
    
    # 强制在 1000 处画一条“截止线”，显得非常严谨 (可选)
    # plt.axvline(1000, color='gray', linestyle=':', alpha=0.5)

    eq_level_g = np.mean(m_g_mean[-500:])
    eq_level_m = np.mean(m_m_mean[-500:])
    overall_eq = (eq_level_g + eq_level_m) / 2  # 取两者的综合基准

    # 2. 增加平衡态基准线 (只在后半段显示，增加专业感)
    # [y, xmin, xmax]
    ax.hlines(y=overall_eq, xmin=1500, xmax=3000, 
              color='gray', linestyle='--', linewidth=1, alpha=0.6, 
              label='Equilibrium Level')
    
    # 3. 如果你想更硬核一点，可以在基准线上方加个微型文字标注
    ax.text(3050, overall_eq, f'|M| ≈ {overall_eq:.2f}', 
            va='center', fontsize=9, color='gray', family='serif')
    
    
    [plt.savefig(f'Ising_Dynamics_Relaxation.{fmt}', dpi=300, bbox_inches='tight') for fmt in ['pdf', 'png']]
   
    
    print("✅ 论文配图已生成! ")    
    plt.show()

if __name__ == "__main__":
    analyze_and_plot()