import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import os
from scipy.optimize import curve_fit

# ==========================================
# 1. 设置顶级期刊风格 (Nature/Science)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stixsans' 
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.linewidth'] = 1.2 
plt.rcParams['xtick.major.width'] = 1.2 
plt.rcParams['ytick.major.width'] = 1.2

def load_data(folder):
    """读取文件夹下所有 .npz 文件并计算平均值和标准差"""
    if not os.path.exists(folder):
        print(f"Error: 文件夹 {folder} 不存在！请检查路径。")
        return None, None, None
        
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npz') and 'avg' not in f])
    print(f"正在从 {folder} 读取 {len(files)} 个文件...")
    
    data_list = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            loaded = np.load(path)
            if 'magnetization' in loaded:
                data_list.append(loaded['magnetization'])
            elif 'history' in loaded:
                 d = loaded['history']
                 if len(d.shape) > 1: data_list.extend(d)
                 else: data_list.append(d)
        except:
            pass
            
    data_matrix = np.array(data_list) # [Runs, Steps]
    if data_matrix.size == 0:
        return None, None, None

    t = np.arange(data_matrix.shape[1])
    m_avg = np.mean(data_matrix, axis=0)
    m_std = np.std(data_matrix, axis=0)
    return t, m_avg, m_std

def exponential_decay(t, tau, A, C):
    """拟合函数: M(t) = A * exp(-t/tau) + C"""
    return A * np.exp(-t / tau) + C

def analyze_and_plot():
    # 文件夹路径
    dir_glauber = "data_dynamics_safe"     
    dir_metro = "metropolis_data_dynamics" 
    
    # 1. 读取数据
    t, m_gl, std_gl = load_data(dir_glauber)
    _, m_me, std_me = load_data(dir_metro)

    if m_gl is None or m_me is None:
        print("数据读取失败，终止绘图。")
        return
    
    # 2. 计算特征时间 tau (拟合前 500 步)
    fit_limit = 500
    try:
        popt_me, _ = curve_fit(exponential_decay, t[:fit_limit], m_me[:fit_limit], p0=[100, 0.8, 0.2])
        tau_me = popt_me[0]
        popt_gl, _ = curve_fit(exponential_decay, t[:fit_limit], m_gl[:fit_limit], p0=[100, 0.8, 0.2])
        tau_gl = popt_gl[0]
    except Exception as e:
        print(f"拟合失败: {e}, 将使用默认值进行演示")
        tau_me, tau_gl = 150.0, 190.0

    print("="*40)
    print(f"📊 物理结果分析")
    print(f"Metropolis tau ≈ {tau_me:.2f}")
    print(f"Glauber    tau ≈ {tau_gl:.2f}")
    print("="*40)

    # 3. 绘图 (切换到面向对象 ax 模式，以便控制 Inset)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # --- 绘制主曲线 ---
    # Metropolis (蓝色系)
    ax.plot(t, m_me, label=f'Metropolis ($\\tau \\approx {tau_me:.1f}$)', color='#1f77b4', lw=2.4, zorder=3)
    ax.fill_between(t, m_me - std_me, m_me + std_me, color='#1f77b4', alpha=0.15, zorder=2, lw=0)

    # Glauber (红色系)
    ax.plot(t, m_gl, label=f'Glauber ($\\tau \\approx {tau_gl:.1f}$)', color='#d62728', lw=2.4, zorder=3)
    ax.fill_between(t, m_gl - std_gl, m_gl + std_gl, color='#d62728', alpha=0.15, zorder=2, lw=0)

    # ==========================================
    # 🌟 新增功能 1: 平衡态基准回归线 (Equilibrium Baseline)---->>>改为标尺箭头
    # ==========================================
    # 计算最后 500 步的平均值作为物理基准
    eq_level_g = np.mean(m_gl[-500:])
    eq_level_m = np.mean(m_me[-500:])
    overall_eq = (eq_level_g + eq_level_m) / 2 

    # 从 x=1500 开始画到结束，展示归宿
    # ax.hlines(y=overall_eq, xmin=1500, xmax=t[-1], 
          # color='black', linestyle='--', linewidth=1.2, alpha=0.5, zorder=10)
    
    # 在线尾添加文字标注
    # ax.text(t[-1]+50, overall_eq, f'$|M|_{{eq}} \\approx {overall_eq:.2f}$', 
            # va='center', ha='left', fontsize=11, color='gray')
            
    ax.annotate(fr'$|M\,|_{{eq}} \approx {overall_eq:.2f}$', 
            xy=(t[-1], overall_eq),       # 箭头尖端位置 (3000, 0.18)
            xytext=(t[-1]-450, overall_eq + 0.2), # 文字位置 (稍微往左上提一点)
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
            fontsize=14, color='black', ha='center')        
            

    # ==========================================
    # 🌟 新增功能 2: 弛豫时间引导线 (Tau Markers)
    # ==========================================
    # 找到 t=tau 时刻对应的 y 值
    y_at_tau_me = m_me[int(tau_me)] if int(tau_me) < len(m_me) else 0
    y_at_tau_gl = m_gl[int(tau_gl)] if int(tau_gl) < len(m_gl) else 0

    # 画垂直虚线 (vlines): 从底画到曲线位置
    ax.vlines(x=tau_me, ymin=0, ymax=y_at_tau_me, colors='#1f77b4', linestyles=':', lw=1.5, alpha=0.8)
    ax.vlines(x=tau_gl, ymin=0, ymax=y_at_tau_gl, colors='#d62728', linestyles=':', lw=1.5, alpha=0.8)

    # ==========================================
    # 🌟 新增功能 3: Inset (局部放大子图)
    # ==========================================
    # [left, bottom, width, height] 这里的 0.5, 0.5 代表右上角区域
    ax_ins = ax.inset_axes([0.48, 0.48, 0.45, 0.45]) 
    
    # 在子图里再画一遍数据
    ax_ins.plot(t, m_me, color='#1f77b4', lw=2)
    ax_ins.plot(t, m_gl, color='#d62728', lw=2)
    
    # *** 这里设置子图的视野 ***
    # 聚焦前 600 步，纵坐标 0.1 到 1.0 (避开底部的长尾)
    ax_ins.set_xlim(0, 600)
    ax_ins.set_ylim(0.1, 1.0)
    
    # 子图美化：精简刻度，加上背景色防止透明干扰
    ax_ins.xaxis.set_major_locator(MaxNLocator(3))
    ax_ins.yaxis.set_major_locator(MaxNLocator(3))
    ax_ins.tick_params(labelsize=10)
    ax_ins.set_facecolor('white') 
    ax_ins.patch.set_alpha(0.9) # 90% 不透明度遮挡后面的主图

    # 添加 "放大镜" 连线效果
    # ax.indicate_inset_zoom(ax_ins, edgecolor="gray", alpha=1)

    # ==========================================
    # 装饰与保存
    # ==========================================
    ax.set_xlabel('Time (MCS)', fontsize=16)
    ax.set_ylabel('Magnetization $|M\,|$', fontsize=16)
    ax.set_xlim(0, 3000)
    ax.set_ylim(bottom=0) # 确保 y 轴从 0 开始
    
    # 图例设置
    # bbox_to_anchor=(x, y) 
    # x=0.95: 靠右侧对齐
    # y=0.45: 放在高度 0.45 的位置 (刚好在子图下方，数据上方)
    legend_header = r'$L=64, T=2.4\,J/k_B$'
    leg = ax.legend(title=legend_header, 
            title_fontsize=13,
            bbox_to_anchor=(0.33, 0.99), # 把图例的右上角钉在主图的 (0.95, 0.45) 处
            fontsize=12, 
            loc='upper right',       # 图例自己的参考点是“右上角”
            
            frameon=False)           # 去掉边框，融入背景
    # 2. 【核心修复】强制修改内部盒子的对齐方式
    # 注意：这行代码必须在 ax.legend() 之后立即执行
    leg._legend_box.align = "left"
    # 保存
    output_filename = 'Ising_Dynamics_Comparison_Final'
    fig.savefig(f'{output_filename}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_filename}.png', dpi=300, bbox_inches='tight')
    
    print(f"✅ 完美配图已保存: {output_filename}.pdf")
    plt.show()

if __name__ == "__main__":
    analyze_and_plot()