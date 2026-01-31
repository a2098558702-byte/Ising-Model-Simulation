import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
import os
import matplotlib.pyplot as plt

def solve_intersection_robust(f1, f2, t_min, t_max):
    """
    鲁棒求交算法：
    1. 先网格扫描，找到变号区间。
    2. 再精确求根。
    """
    # 1. 网格扫描 (比如扫 5000 个点)
    t_scan = np.linspace(t_min, t_max, 5000)
    diff = f1(t_scan) - f2(t_scan)
    
    # 2. 寻找符号变化的点 (Sign change)
    # signs[i] 为 True 表示 diff[i] 和 diff[i-1] 符号不同
    signs = np.sign(diff[:-1]) != np.sign(diff[1:])
    change_indices = np.where(signs)[0]
    
    roots = []
    for idx in change_indices:
        # 锁定一个小区间 [t_left, t_right]
        t_left = t_scan[idx]
        t_right = t_scan[idx+1]
        
        try:
            # 在小区间内精确求根
            root = brentq(lambda x: f1(x) - f2(x), t_left, t_right)
            # 过滤掉不在物理范围内的伪解 (比如样条震荡产生的)
            if 2.1 < root < 2.4: 
                roots.append(root)
        except:
            pass
            
    return roots

def calculate_precise_tc_robust():
    input_folder = 'data_ultimate_u4'
    L_list = [16, 32, 48, 64, 80, 128]
    
    # 加载数据的逻辑不变...
    splines = {}
    valid_L = []
    
    # === 1. 数据读取与样条构建 (保持原逻辑) ===
    print(f"{'='*60}")
    print(f"🔧 正在构建样条函数...")
    for L in L_list:
        try:
            # 优先读标准化数据
            path = os.path.join(input_folder, f'Standardized_Binder_L{L}.npz')
            if not os.path.exists(path): path = os.path.join(input_folder, f'u4_L{L}.npz')
            
            if not os.path.exists(path): continue
            
            data = np.load(path)
            if 'T_raw' in data: t, u4 = data['T_raw'], data['U4_raw']
            elif 'T' in data: t, u4 = data['T'], data['u4'] if 'u4' in data else data['U4']
            else: continue
            
            idx = np.argsort(t)
            t, u4 = t[idx], u4[idx]
            
            w = None
            if 'U4_err' in data: w = 1/(data['U4_err'][idx] + 1e-10)
            
            # 【重点】这里 s 稍微设大一点点，防止过拟合造成的假交点
            spl = UnivariateSpline(t, u4, w=w, k=3, s=len(t)) 
            splines[L] = spl
            valid_L.append(L)
            
            # 打印数据范围，帮助debug
            print(f"  L={L}: T range [{min(t):.3f}, {max(t):.3f}]")
            
        except Exception as e:
            print(f"❌ L={L} 读数失败: {e}")

    # === 2. 鲁棒求交 ===
    print(f"\n🔍 开始全域搜索交点 (Range: 2.1 - 2.5)...")
    print(f"-"*50)
    print(f"{'Pairs':<15} | {'Found Tc'}")
    print(f"-"*50)

    found_tcs = []

    for i in range(len(valid_L)-1):
        L1 = valid_L[i]
        L2 = valid_L[i+1]
        
        # 使用鲁棒求解器，范围放宽到 2.1 到 2.5
        roots = solve_intersection_robust(splines[L1], splines[L2], 2.1, 2.5)
        
        if len(roots) == 0:
            print(f"{L1} vs {L2:<3}    | ❌ 未找到 (请检查数据是否相交)")
            
            # Debug: 如果找不到，画出差值图看看
            # plt.figure()
            # tx = np.linspace(2.1, 2.5, 100)
            # plt.plot(tx, splines[L1](tx) - splines[L2](tx))
            # plt.title(f"Diff {L1}-{L2}")
            # plt.grid(); plt.show()
            
        else:
            # 如果有多个交点，取最接近 2.269 的那个
            best_root = min(roots, key=lambda x: abs(x - 2.269))
            print(f"{L1} vs {L2:<3}    | {best_root:.6f}")
            found_tcs.append(best_root)

    # === 3. 统计输出 ===
    if found_tcs:
        found_tcs = np.array(found_tcs)
        
        # 剔除 L=16 (通常不准)
        if len(found_tcs) > 2:
            final_tcs = found_tcs[1:]
            note = "(剔除 L=16)"
        else:
            final_tcs = found_tcs
            note = "(全量)"
            
        mean_tc = np.mean(final_tcs)
        std_tc = np.std(final_tcs)
        
        print(f"\n{'-'*50}")
        print(f"✅ 最终结果 {note}:")
        print(f"Tc = {mean_tc:.5f} ± {std_tc:.5f}")
        print(f"{'-'*50}")
        
        # 保存结果
        with open(os.path.join(input_folder, 'Tc_Robust_Result.txt'), 'w') as f:
            f.write(f"Tc_Mean = {mean_tc:.6f}\n")
            f.write(f"Tc_Std = {std_tc:.6f}\n")
            f.write(f"All_Roots = {found_tcs.tolist()}\n")

if __name__ == "__main__":
    calculate_precise_tc_robust()