import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    # --- 核心修改点：强制数学公式也使用无衬线体 ---
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})
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
    "axes.unicode_minus": False
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
# 4. 参数配置区
# ==========================================
ins_params_a = {
    "width": "38%", 
    "height": "35%", 
    "fit_steps": 1000,
    "bbox": (0.48, 0.40, 1, 1) 
}

# 数据加载
m_24_glau_avg, m_24_glau_std = aggregate_runs("data_dynamics_safe", "run_*.npz", "magnetization")
m_24_metro_avg, m_24_metro_std = aggregate_runs("metropolis_data_dynamics", "run_*.npz", "magnetization")
m_227_metro_avg, m_227_metro_std = aggregate_runs("data_dynamics_T2.27", "Metropolis_run*.npz", "m_history")
m_227_glau_avg, m_227_glau_std = aggregate_runs("data_dynamics_T2.27", "Glauber_run*.npz", "m_history")

# ==========================================
# 5. 绘图执行
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
plt.subplots_adjust(wspace=0.28)

# --- Panel A ---
t1 = np.arange(len(m_24_glau_avg))
limit_a = 3000
ax1.plot(t1[:limit_a], m_24_glau_avg[:limit_a], color=color_g, lw=1.6, label='Glauber', zorder=2)
ax1.fill_between(t1[:limit_a], m_24_glau_avg[:limit_a]-m_24_glau_std[:limit_a], 
                 m_24_glau_avg[:limit_a]+m_24_glau_std[:limit_a], color=color_g, alpha=0.12)
ax1.plot(t1[:limit_a], m_24_metro_avg[:limit_a], color=color_m, lw=1.6, label='Metropolis', zorder=3)
ax1.fill_between(t1[:limit_a], m_24_metro_avg[:limit_a]-m_24_metro_std[:limit_a], 
                 m_24_metro_avg[:limit_a]+m_24_metro_std[:limit_a], color=color_m, alpha=0.12)

ax1.axhline(y=0.15, color='gray', ls='--', lw=1, alpha=0.6)

# [修改点] 这里加上了括号: \mathbf{(a)}
ax1.set_title(r"$\mathbf{(a)} \quad T = 2.40 \, J/k_B, \, L = 64$", loc='left', pad=12, fontweight='bold')

ax1.set_xlim(-100, 3100) 
add_clean_inset(ax1, t1, m_24_metro_avg, m_24_glau_avg, 0.15, ins_params_a)

# --- Panel B ---
t2 = np.arange(len(m_227_glau_avg))
limit_b = 10000
ax2.plot(t2[:limit_b], m_227_glau_avg[:limit_b], color=color_g, lw=1.4, label='Glauber')
ax2.fill_between(t2[:limit_b], m_227_glau_avg[:limit_b]-m_227_glau_std[:limit_b], 
                 m_227_glau_avg[:limit_b]+m_227_glau_std[:limit_b], color=color_g, alpha=0.15)
ax2.plot(t2[:limit_b], m_227_metro_avg[:limit_b], color=color_m, lw=1.4, label='Metropolis')
ax2.fill_between(t2[:limit_b], m_227_metro_avg[:limit_b]-m_227_metro_std[:limit_b], 
                 m_227_metro_avg[:limit_b]+m_227_metro_std[:limit_b], color=color_m, alpha=0.15)

ax2.axhline(y=0.60, color='gray', ls='--', lw=1, alpha=0.6)

# [修改点] 这里加上了括号: \mathbf{(b)}
ax2.set_title(r"$\mathbf{(b)} \quad T = 2.27 \, J/k_B, \, L = 64$", loc='left', pad=12, fontweight='bold')

ax2.set_xlim(-300, 10300) 

# --- 统一装饰 ---
ax1.set_ylabel(r"Magnetization $\langle |m\,| \rangle$")
for ax in [ax1, ax2]:
    ax.set_xlabel("Time (MCS)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, ls=':', alpha=0.3)
    ax.legend(frameon=False, loc='upper right' if ax==ax1 else 'lower right')
    for spine in ax.spines.values(): spine.set_linewidth(1.3)

# ==========================================
# 5. 绘图执行 (修正字体一致性)
# ==========================================
# ... 之前的绘图代码 ...

# --- Panel A ---
# [修改点] 将 (a) 移出 $ $，利用 fontweight='bold' 实现全局字体加粗
ax1.set_title("(a) " + r"$T = 2.40 \, J/k_B, \, L = 64$", loc='left', pad=12, fontweight='bold')

# --- Panel B ---
# [修改点] 同样处理 (b)
ax2.set_title("(b) " + r"$T = 2.27 \, J/k_B, \, L = 64$", loc='left', pad=12, fontweight='bold')

# ... 剩下的装饰代码 ...

# 保存与显示
plt.savefig("Ising_Dynamics.pdf", bbox_inches='tight')
plt.savefig("Ising_Dynamics.png", bbox_inches='tight')
plt.show()