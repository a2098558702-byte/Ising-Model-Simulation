import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

# ================= 1. 路径配置 (请再次确认绝对路径) =================
# 建议直接保留你刚才测试通过的那个路径
# 注意：路径字符串前加 r

PATH_T2_4_GL = r"data_dynamics_safe"           
PATH_T2_4_ME = r"metropolis_data_dynamics"     
PATH_T2_27   = r"data_dynamics_T2.27"          

# ================= 2. 参数配置 =================
M_EQ_2_4 = 0.156  
M_EQ_2_27 = 0.595 

FIT_RANGE_2_4 = (0, 500)
FIT_RANGE_2_27 = (0, 500) 

# ================= 3. 核心逻辑函数 =================

def load_and_average(folder_path, filename_pattern, file_range, key_name, algo_name):
    print(f"--- 正在处理 {algo_name} ---")
    all_runs = []
    missing_count = 0
    
    for i in file_range:
        # 1. 构建基础文件名 (无后缀)
        if "XXX" in filename_pattern:
            fname_base = filename_pattern.replace("XXX", f"{i:03d}") # run_001
        else:
            fname_base = filename_pattern.replace("XX", f"{i:02d}")  # run00
            
        base_path = os.path.join(folder_path, fname_base)
        
        # 2. 智能探测后缀 (.npz 优先，然后 .npy)
        if os.path.exists(base_path + ".npz"):
            full_path = base_path + ".npz"
        elif os.path.exists(base_path + ".npy"):
            full_path = base_path + ".npy"
        else:
            # 找不到文件
            missing_count += 1
            continue
            
        # 3. 读取数据
        try:
            # np.load 可以读取 .npy 和 .npz
            data = np.load(full_path, allow_pickle=True)
            
            # 4. 提取数据 (核心修复点)
            m_seq = None
            
            # 情况 A: .npz 文件 (类似字典)
            if hasattr(data, 'files'): 
                if key_name in data:
                    m_seq = data[key_name]
            # 情况 B: .npy 文件 (可能存了字典，也可能直接是数组)
            elif isinstance(data.item(), dict):
                if key_name in data.item():
                    m_seq = data.item()[key_name]
            else:
                # 也许直接就是数组？
                pass

            if m_seq is not None:
                all_runs.append(np.abs(m_seq)) # 确保取绝对值
            else:
                print(f"⚠️ 文件 {full_path} 中找不到键值 '{key_name}'")

        except Exception as e:
            print(f"读取错误 {full_path}: {e}")
            missing_count += 1

    if len(all_runs) == 0:
        raise ValueError(f"❌ 在 {folder_path} 下没读到任何数据！请检查路径或文件名格式。")

    print(f"✅ 成功加载: {len(all_runs)} 个文件 (缺失: {missing_count})")
    
    # 截断并平均
    min_len = min([len(x) for x in all_runs])
    matrix = np.array([x[:min_len] for x in all_runs])
    m_avg = np.mean(matrix, axis=0)
    
    return m_avg

def fit_tau(m_data, m_eq, fit_range, label):
    start, end = fit_range
    t = np.arange(start, end)
    m_segment = m_data[start:end]
    
    delta_m = np.abs(m_segment - m_eq)
    valid_mask = delta_m > 1e-6
    t_clean = t[valid_mask]
    log_delta_m = np.log(delta_m[valid_mask])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_clean, log_delta_m)
    
    tau = -1.0 / slope
    tau_err = (tau ** 2) * std_err
    r_squared = r_value ** 2
    
    return t_clean, log_delta_m, slope, intercept, tau, tau_err, r_squared

# ================= 4. 主程序 =================

def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- T = 2.4 ---
    print("\n>>>> 处理 T = 2.4 数据...")
    # 注意：文件名格式 run_XXX (对应 run_001.npz)
    m_avg_metro_24 = load_and_average(PATH_T2_4_ME, "run_XXX", range(1, 201), 'magnetization', "Metropolis T2.4")
    m_avg_gl_24    = load_and_average(PATH_T2_4_GL, "run_XXX", range(1, 201), 'magnetization', "Glauber T2.4")
    
    res_m_24 = fit_tau(m_avg_metro_24, M_EQ_2_4, FIT_RANGE_2_4, "Me T2.4")
    res_g_24 = fit_tau(m_avg_gl_24,    M_EQ_2_4, FIT_RANGE_2_4, "Gl T2.4")
    
    ax = axes[0]
    ax.plot(res_m_24[0], res_m_24[2]*res_m_24[0] + res_m_24[3], 'b-', label=f'Metro $\\tau={res_m_24[4]:.1f}$')
    ax.scatter(res_m_24[0], res_m_24[1], s=2, c='b', alpha=0.3)
    ax.plot(res_g_24[0], res_g_24[2]*res_g_24[0] + res_g_24[3], 'r-', label=f'Glauber $\\tau={res_g_24[4]:.1f}$')
    ax.scatter(res_g_24[0], res_g_24[1], s=2, c='r', alpha=0.3)
    ax.set_title("T=2.4 Relaxation")
    ax.legend()

    # --- T = 2.27 ---
    print("\n>>>> 处理 T = 2.27 数据...")
    # 注意：文件名格式 Metropolis_runXX (对应 Metropolis_run00.npz/npy)
    # 这里的 Key 是 'm_history'
    m_avg_metro_227 = load_and_average(PATH_T2_27, "Metropolis_runXX", range(0, 200), 'm_history', "Metropolis T2.27")
    m_avg_gl_227    = load_and_average(PATH_T2_27, "Glauber_runXX",    range(0, 200), 'm_history', "Glauber T2.27")
    
    res_m_227 = fit_tau(m_avg_metro_227, M_EQ_2_27, FIT_RANGE_2_27, "Me T2.27")
    res_g_227 = fit_tau(m_avg_gl_227,    M_EQ_2_27, FIT_RANGE_2_27, "Gl T2.27")
    
    ax = axes[1]
    ax.plot(res_m_227[0], res_m_227[2]*res_m_227[0] + res_m_227[3], 'b-', label=f'Metro $\\tau={res_m_227[4]:.1f}$')
    ax.scatter(res_m_227[0], res_m_227[1], s=2, c='b', alpha=0.3)
    ax.plot(res_g_227[0], res_g_227[2]*res_g_227[0] + res_g_227[3], 'r-', label=f'Glauber $\\tau={res_g_227[4]:.1f}$')
    ax.scatter(res_g_227[0], res_g_227[1], s=2, c='r', alpha=0.3)
    ax.set_title("T=2.27 Relaxation")
    ax.legend()
    
    plt.tight_layout()
    plt.show()

    # --- 输出结论 ---
    print("\n" + "="*40)
    print(" >>> 论文填空数据 (直接复制) <<<")
    print("="*40)
    print(f"[T=2.40]")
    print(f"  Metropolis Tau = {res_m_24[4]:.2f} ± {res_m_24[5]:.2f} (R2={res_m_24[6]:.4f})")
    print(f"  Glauber    Tau = {res_g_24[4]:.2f} ± {res_g_24[5]:.2f} (R2={res_g_24[6]:.4f})")
    eff_24 = (res_g_24[4] - res_m_24[4]) / res_g_24[4] * 100
    print(f"  效率提升: {eff_24:.1f}%")
    
    print(f"\n[T=2.27]")
    print(f"  Metropolis Tau = {res_m_227[4]:.2f} ± {res_m_227[5]:.2f} (R2={res_m_227[6]:.4f})")
    print(f"  Glauber    Tau = {res_g_227[4]:.2f} ± {res_g_227[5]:.2f} (R2={res_g_227[6]:.4f})")
    eff_227 = (res_g_227[4] - res_m_227[4]) / res_g_227[4] * 100
    print(f"  效率提升: {eff_227:.1f}%")

if __name__ == "__main__":
    main()