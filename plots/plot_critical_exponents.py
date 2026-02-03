import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.optimize import curve_fit

# ... (保持之前的样式设置不变) ...
plt.rcParams.update({
    'font.size': 10, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'stixsans', 'xtick.direction': 'in', 'ytick.direction': 'in',
})

def load_latest_data():
    # ... (保持之前的读取逻辑不变) ...
    all_dirs = glob.glob("data_critical_fit_*")
    if not all_dirs: raise FileNotFoundError("❌ 未找到数据文件夹")
    latest_dir = max(all_dirs, key=os.path.getmtime)
    print(f"📂 读取数据: {latest_dir}")
    
    files = glob.glob(f"{latest_dir}/*.npz")
    data_list = []
    for f in files:
        d = np.load(f)
        L = float(d['L'])
        T = float(d['T'])
        beta = 1.0/T
        m_abs = float(d['m_abs_mean'])
        m_sq = float(d['m_sq_mean'])
        
        magnetization = m_abs / (L**2)
        susceptibility = beta * (m_sq - m_abs**2) / (L**2)
        
        data_list.append({'L': L, 'M': magnetization, 'Chi': susceptibility})
    
    data_list.sort(key=lambda x: x['L'])
    return (np.array([x['L'] for x in data_list]), 
            np.array([x['M'] for x in data_list]), 
            np.array([x['Chi'] for x in data_list]))

def linear_fit(x, k, b):
    return k * x + b

def get_fit_with_error(x, y):
    # 核心修改：这里不仅返回斜率，还返回误差
    popt, pcov = curve_fit(linear_fit, x, y)
    slope = popt[0]
    # pcov 的对角线是方差，开根号就是标准误差 (Standard Error)
    perr = np.sqrt(np.diag(pcov))
    slope_err = perr[0] 
    r_squared = 1 - (np.sum((y - linear_fit(x, *popt))**2) / np.sum((y - np.mean(y))**2))
    return slope, slope_err, r_squared, popt

def main():
    L, M, Chi = load_latest_data()[:3] # 只取前三个变量
    
    # --- 1. 拟合 Magnetization (Beta/Nu) ---
    x_m = np.log(L)
    y_m = np.log(M)
    slope_m, err_m, r2_m, _ = get_fit_with_error(x_m, y_m)
    
    # --- 2. 拟合 Susceptibility (Gamma/Nu) ---
    x_chi = np.log(L)
    y_chi = np.log(Chi)
    slope_chi, err_chi, r2_chi, _ = get_fit_with_error(x_chi, y_chi)

    # --- 3. 打印“金标准”数据 ---
    print("\n" + "="*50)
    print("💎 论文数据速查表 (请直接复制这些数字)")
    print("="*50)
    
    # 辅助函数：格式化为 0.123(4)
    def fmt(val, err):
        # 取绝对值（因为 M 的斜率是负的，但指数比是正的）
        val = abs(val) 
        if err == 0: return f"{val:.4f}"
        import math
        order = int(math.floor(math.log10(err)))
        decimals = -order + 1 # 保留两位误差有效数字
        if decimals < 0: decimals = 0
        return f"{val:.{decimals}f}({int(err * 10**decimals)})"

    beta_nu_str = fmt(slope_m, err_m)
    gamma_nu_str = fmt(slope_chi, err_chi)

    print(f"拟合原始数据:")
    print(f"  Slope M   = {slope_m:.6f} ± {err_m:.6f}")
    print(f"  Slope Chi = {slope_chi:.6f} ± {err_chi:.6f}")
    print("-" * 50)
    print(f"LaTeX 填空推荐格式:")
    print(f"  beta/nu  = {beta_nu_str}")
    print(f"  gamma/nu = {gamma_nu_str}")
    print("="*50)

if __name__ == "__main__":
    main()