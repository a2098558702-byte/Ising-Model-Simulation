import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ==========================================
# 0. 单栏学术绘图标准设置 (The "Nature" Standard)
# ==========================================
# 定义单栏宽度 (PRL/Nature 标准约为 86mm ~ 3.4 inches)
SINGLE_COL_WIDTH = 3.4 
GOLDEN_RATIO = 0.618 + 0.1 # 稍微调高一点高度，给 Inset 留空间

plt.rcParams.update({
    'figure.figsize': (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.75), # 4:3 比例
    'font.size': 8,              # 全局字体 8pt
    'axes.titlesize': 8,         # 标题
    'axes.labelsize': 9,         # 轴标签 (略大)
    'xtick.labelsize': 8,        # 刻度
    'ytick.labelsize': 8,
    'legend.fontsize': 7,        # 图例 (小一点，省空间)
    'lines.linewidth': 1.0,      # 线宽 (小图 1.0 足够清晰)
    'lines.markersize': 3,       # 散点大小
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 0.8,       # 边框变细
})

# ==========================================
# 1. 准备工作
# ==========================================
L_list = [16, 32, 48, 64, 80, 128]
save_dir = 'data_ultimate_u4'

# 创建画布 (尺寸已由 rcParams 控制)
fig, ax = plt.subplots(dpi=300)

# 生成渐变色
colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_list)))

# ==========================================
# 【保持不变】创建局部放大子图
# ==========================================
# 位置完全保持你设定的 [0.27, 0.16, 0.38, 0.43]
ax_ins = ax.inset_axes([0.27, 0.16, 0.38, 0.43])

# ==========================================
# 2. 循环加载数据并绘图
# ==========================================
print("正在绘制 U4 曲线 (单栏适配版)...")

for i, L in enumerate(L_list):
    try:
        # 加载数据
        filename = f"{save_dir}/u4_L{L}.npz"
        data = np.load(filename)
        T_data = data['T']
        u4_data = data['u4'] 
        
        # 滤波 (保持不变)
        window_len = min(11, len(T_data)) 
        if window_len % 2 == 0: window_len -= 1
        
        if len(T_data) > 15:
            u4_smooth = savgol_filter(u4_data, window_length=window_len, polyorder=3)
        else:
            u4_smooth = u4_data

        # --- 绘图 (修改点：线宽变细，点变小) ---
        
        # 主图曲线：linewidth 改为 1.0 (原 1.8)
        ax.plot(T_data, u4_smooth, 
                label=f'L={L}', 
                color=colors[i], 
                linewidth=1.0,  # <--- 修改
                alpha=0.9)
                
        # 主图散点：s 改为 4 (原 15)，透明度降低
        ax.scatter(T_data, u4_data, color=colors[i], s=4, alpha=0.2, marker='o', edgecolors='none') # <--- 修改

        # 子图曲线：linewidth 改为 1.2 (原 2.0，稍微比主线粗一点点突出显示即可)
        ax_ins.plot(T_data, u4_smooth, color=colors[i], linewidth=1.2, alpha=0.9) # <--- 修改

    except FileNotFoundError:
        print(f"⚠️ 跳过 L={L}")

# ==========================================
# 3. 装饰图表
# ==========================================

# 标签字体大小由 rcParams 全局控制 (约为 9pt)，不再硬编码 14
plt.xlabel('Temperature $T$ ($J/k_B$)') 
plt.ylabel('Binder Cumulant $U_4$')

# 图例：去除边框，紧凑布局
plt.legend(frameon=False, loc='best', handlelength=1.2) 

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5) 

# ==========================================
# 4. 坐标轴微调
# ==========================================
ax.set_xlim(T_data.min(), T_data.max()) 
ax.set_ylim(0, 0.7)

# 隐藏重叠刻度 (The Clean Corner)
yticks = ax.yaxis.get_major_ticks()
if len(yticks) > 0:
    yticks[0].label1.set_visible(False)

# ==========================================
# 【保持不变】装饰局部放大子图
# ==========================================
x1, x2 = 2.258, 2.280 
y1, y2 = 0.585, 0.64 

ax_ins.set_xlim(x1, x2)
ax_ins.set_ylim(y1, y2)

# 精简子图刻度 (保持你的 MaxNLocator)
from matplotlib.ticker import MaxNLocator
ax_ins.xaxis.set_major_locator(MaxNLocator(3))
ax_ins.yaxis.set_major_locator(MaxNLocator(3))
# 刻度线变细
ax_ins.tick_params(width=0.6) 

# 连接线变细
ax.indicate_inset_zoom(ax_ins, edgecolor="gray", alpha=0.5, linewidth=0.8)

# ==========================================
# 保存
# ==========================================
# 这里的 pad_inches=0.02 对于单栏图非常重要，能最大化利用空间
[plt.savefig(f'Ising_U4_Intersection_SingleCol.{fmt}', dpi=300, bbox_inches='tight', pad_inches=0.02) for fmt in ['pdf', 'png']]

plt.show()