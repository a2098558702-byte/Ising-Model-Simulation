import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ==========================================
# 1. 顶刊级全局配置 (无衬线风格)
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,          
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "axes.unicode_minus": False,
    # --- 强制数学公式使用无衬线体 ---
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

color_g = '#E64B35' 
color_m = '#4DBBD5' 

# ==========================================
# 2. 核心数据聚合函数
# ==========================================
def aggregate_runs(folder, pattern, var_name, num_runs=200):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files: return None, None
    data_stack = []
    for f in files[:num_runs]:
        with np.load(f) as d:
            data_stack.append(np.abs(d[var_name]))
    data_stack = np.array(data_stack)
    return np.mean(data_stack, axis=0), np.std(data_stack, axis=0)

# ==========================================
# 3. 增强版 Inset 函数
# ==========================================
def add_clean_inset(ax, t, avg_m, avg_g, baseline, params):
    steps = params["fit_steps"]
    
    # 使用 bbox_to_anchor 进行绝对定位
    ax_ins = inset_axes(ax, 
                        width=params["width"], 
                        height=params["height"], 
                        loc='lower left', 
                        bbox_to_anchor=params["bbox"], 
                        bbox_transform=ax.transAxes, 
                        borderpad=0)
    
    def get_log_data(avg):
        t_slice = t[:steps]
        diff = avg[:steps] - baseline
        mask = diff > 1e-6 
        return t_slice[mask], np.log(diff[mask])

    t_m_log, val_m_log = get_log_data(avg_m)
    t_g_log, val_g_log = get_log_data(avg_g)

    ax_ins.plot(t_g_log, val_g_log, color=color_g, lw=1.2, ls='--')
    ax_ins.plot(t_m_log, val_m_log, color=color_m, lw=1.4)
    
    ax_ins.set_xticks([]) 
    ax_ins.set_ylabel(r"$\ln(\Delta m)$", fontsize=9, labelpad=1)
    ax_ins.tick_params(axis='y', labelsize=8, length=2, pad=2)
    for spine in ax_ins.spines.values(): spine.set_linewidth(0.8)

# ==========================================
# 4. 数据加载 (仅加载 T=2.40)
# ==========================================
# 注意：请确保这些文件夹路径与你本地一致
m_24_glau_avg, m_24_glau_std = aggregate_runs("data_dynamics_safe", "run_*.npz", "magnetization")
m_24_metro_avg, m_24_metro_std = aggregate_runs("metropolis_data_dynamics", "run_*.npz", "magnetization")

# ==========================================
# 5. 绘图执行 (单张图版)
# ==========================================
ins_params_a = {
    "width": "38%", 
    "height": "35%", 
    "fit_steps": 1000,
    "bbox": (0.48, 0.40, 1, 1) 
}

# 修改点：改为 (1, 1)，调整 figsize 为更适合单图的比例 (6, 4.6)
fig, ax1 = plt.subplots(1, 1, figsize=(6, 4.6))

# --- 绘图逻辑 ---
if m_24_glau_avg is not None and m_24_metro_avg is not None:
    t1 = np.arange(len(m_24_glau_avg))
    limit_a = 3000
    
    # 主曲线
    ax1.plot(t1[:limit_a], m_24_glau_avg[:limit_a], color=color_g, lw=1.6, label='Glauber', zorder=2)
    ax1.fill_between(t1[:limit_a], m_24_glau_avg[:limit_a]-m_24_glau_std[:limit_a], 
                     m_24_glau_avg[:limit_a]+m_24_glau_std[:limit_a], color=color_g, alpha=0.12)
    
    ax1.plot(t1[:limit_a], m_24_metro_avg[:limit_a], color=color_m, lw=1.6, label='Metropolis', zorder=3)
    ax1.fill_between(t1[:limit_a], m_24_metro_avg[:limit_a]-m_24_metro_std[:limit_a], 
                     m_24_metro_avg[:limit_a]+m_24_metro_std[:limit_a], color=color_m, alpha=0.12)

    ax1.axhline(y=0.15, color='gray', ls='--', lw=1, alpha=0.6)

    # [核心修改点] 去掉 (a)，只保留物理参数标题
    # fontweight='bold' 让标题加粗，显眼
    ax1.set_title(r"$T = 2.40 \, J/k_B, \, L = 64$", loc='left', pad=12, fontweight='bold')

    ax1.set_xlim(-100, 3100) 
    
    # 添加插图
    add_clean_inset(ax1, t1, m_24_metro_avg, m_24_glau_avg, 0.15, ins_params_a)

    # 装饰
    ax1.set_ylabel(r"Magnetization $\langle |m\,| \rangle$")
    ax1.set_xlabel("Time (MCS)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, ls=':', alpha=0.3)
    ax1.legend(frameon=False, loc='upper right')
    for spine in ax1.spines.values(): spine.set_linewidth(1.3)
else:
    print("Warning: Data files not found. Check your folder paths.")

# 保存 PNG
plt.tight_layout()
plt.savefig("Ising_Dynamics_Single.png", bbox_inches='tight', dpi=300)
plt.show()