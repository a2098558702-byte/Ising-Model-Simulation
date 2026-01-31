import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# 使用 LaTeX 渲染数学符号
# ax.set_xlabel(r'Temperature $T$ ($J/k_B$)', fontsize=14)
# ax.set_ylabel(r'Binder Cumulant $U_4$', fontsize=14)
# ==========================================
# 1. 准备工作
# ==========================================
L_list = [16, 32, 48, 64, 80, 128]
save_dir = 'data_ultimate_u4'

# 物理绘图标准
plt.rcParams['mathtext.fontset'] = 'stix'

# 设置画图风格 (保持之前约定的 Arial/无衬线风格，清晰度最高)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stixsans' 
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# 创建画布
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

# 生成渐变色 (从蓝紫色到黄色，区分度高)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_list)))


# ==========================================
# 【新增】创建局部放大子图 (Inset Axes)
# ==========================================
# [left, bottom, width, height] 是相对于主图的归一化坐标 (0-1)
# 这里放在左下角空白处
ax_ins = ax.inset_axes([0.27, 0.16, 0.38, 0.43])


# ==========================================
# ==========================================
# 2. 循环加载数据并绘图 (去噪版)
# ==========================================
print("正在绘制 U4 曲线 (启用 Savitzky-Golay 滤波)...")

for i, L in enumerate(L_list):
    try:
        # 1. 加载数据
        filename = f"{save_dir}/u4_L{L}.npz"
        data = np.load(filename)
        T_data = data['T']
        u4_data = data['u4'] 
        
        # --- 关键修改：使用 Savitzky-Golay 滤波器去噪 ---
        # window_length: 窗口长度，必须是奇数。越大越平滑，但太大会失真。
        # polyorder: 多项式阶数。通常 2 或 3。
        
        # 根据你的数据点密度自动调整窗口
        window_len = min(11, len(T_data)) 
        if window_len % 2 == 0: window_len -= 1 # 保证是奇数
        
        if len(T_data) > 15: # 数据点够多才滤波
            u4_smooth = savgol_filter(u4_data, window_length=window_len, polyorder=3)
        else:
            u4_smooth = u4_data

        # 2. 绘图
        # 画平滑后的线
        ax.plot(T_data, u4_smooth, 
                label=f'L={L}', 
                color=colors[i], 
                linewidth=1.8, # 线稍微粗一点更显眼
                alpha=0.9)
                
        # (强烈建议) 画出原始数据点，以示诚实
        # 用很小的点、很淡的颜色，证明你的线没有瞎画
        ax.scatter(T_data, u4_data, color=colors[i], s=15, alpha=0.3, marker='o', edgecolors='none')


        # ==========================================
        # 【新增】子图：同步画平滑线
        # ==========================================
        # 子图里只画线，不画散点，为了看清交点
        ax_ins.plot(T_data, u4_smooth, color=colors[i], linewidth=2.0, alpha=0.9)

    except FileNotFoundError:
        print(f"⚠️ 警告: 找不到文件 {filename}，跳过 L={L}")

# ==========================================
# 3. 装饰图表 (严格执行你的要求)
# ==========================================

# X轴标签：改为温度 T
plt.xlabel('Temperature $T$ ($J/k_B$)', fontsize=14)

# Y轴标签：改为 Binder Cumulant U4
plt.ylabel('Binder Cumulant $U_4$', fontsize=14)

# 一般不用标题，而是写在图的下面
# plt.title(...) 

# 图例设置 (U4图线容易在右边发散，放在左下角或最佳位置可能更好，但这里遵照要求放右上)
# 建议：如果右上角挡住了线，可以改成 loc='lower left'
plt.legend(fontsize=10, loc='best', frameon=False) 

# 虚线网格更优雅
plt.grid(True, alpha=0.3, linestyle='--') 

# ==========================================
# 4. 坐标轴微调 (让图更好看)
# ==========================================
# 自动调整 X 轴范围，聚焦在数据区域
# 假设你在 2.2 - 2.4 之间测的，这样会自动切掉多余空白
ax.set_xlim(T_data.min(), T_data.max()) 
ax.set_ylim(0, 0.7) # U4 理论最大值是 2/3 (0.666...)，设置到 0.7 刚好

# 5. 顶级论文技巧：解决原点标签重叠 (The Clean Corner)
# ==========================================
# 获取 y 轴的所有主刻度对象
yticks = ax.yaxis.get_major_ticks()

# 检查一下是否有刻度，如果有，就对第一个下手
if len(yticks) > 0:
    # yticks[0] 是最下面的那个刻度 (对应 0.0)
    # label1 是左侧的标签文本
    # set_visible(False) 把它变透明
    yticks[0].label1.set_visible(False)

# (可选) 如果你想隐藏 x 轴最左边的 (2.22) 保留 y 轴的 0.0，就用下面这段
# xticks = ax.xaxis.get_major_ticks()
# if len(xticks) > 0:
#     xticks[0].label1.set_visible(False)


# ==========================================
# 【新增】装饰局部放大子图
# ==========================================
# 1. 设置聚焦区域 (根据你的数据相变点 T_c ≈ 2.269 手动微调)
# 这里的范围要非常精准地框住交点
x1, x2 = 2.258, 2.280  # T 的放大范围
y1, y2 = 0.585, 0.64  # U4 的放大范围

ax_ins.set_xlim(x1, x2)
ax_ins.set_ylim(y1, y2)

# 2. 精简子图刻度 (防止太挤)
from matplotlib.ticker import MaxNLocator
ax_ins.xaxis.set_major_locator(MaxNLocator(3)) # X轴最多显示3个刻度数字
ax_ins.yaxis.set_major_locator(MaxNLocator(3)) # Y轴最多显示3个刻度数字
ax_ins.tick_params(labelsize=9) # 刻度字体小一点

# 3. 添加连接线 (视觉引导)
# loc1, loc2 指的是连接主图区域的哪两个角 (1=右上, 2=左上, 3=左下, 4=右下)
# mark_inset(ax, ax_ins, loc1=2, loc2=4, fc="none", ec="0.4", ls='--', lw=0.8)

ax.indicate_inset_zoom(ax_ins, edgecolor="gray", alpha=0.5, linewidth=1.0)


# 保存图片
[plt.savefig(f'Ising_U4_Intersection.{fmt}', dpi=300, bbox_inches='tight') for fmt in ['pdf', 'png']]

plt.show()