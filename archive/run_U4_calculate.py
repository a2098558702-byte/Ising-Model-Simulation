import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
import os

def calculate_precise_tc_statistics():
    input_folder = 'data_ultimate_u4'
    # 你的 L 列表
    L_list = [16, 32, 48, 64, 80, 128]
    
    # 存储重建的样条函数
    splines = {}
    valid_L = []
    
    print(f"{'='*60}")
    print(f"🧮 正在执行多重交点分析 (Intersection Statistics)...")
    print(f"{'='*60}")

    # 1. 加载数据并重建高精度样条
    for L in L_list:
        # 尝试读取标准化数据（优先）或原始数据
        try:
            path = os.path.join(input_folder, f'Standardized_Binder_L{L}.npz')
            if not os.path.exists(path):
                # 回退方案
                path = os.path.join(input_folder, f'u4_L{L}.npz')
            
            if not os.path.exists(path):
                print(f"⚠️ 缺失 L={L} 数据，跳过")
                continue
                
            data = np.load(path)
            
            # 读取 T 和 U4
            if 'T_raw' in data: 
                t, u4 = data['T_raw'], data['U4_raw']
            elif 'T' in data:
                t, u4 = data['T'], data['u4'] if 'u4' in data else data['U4']
            else:
                continue
            
            # 按 T 排序
            idx = np.argsort(t)
            t, u4 = t[idx], u4[idx]
            
            # 读取误差用于加权（如果有）
            w = None
            if 'U4_err' in data:
                err = data['U4_err'][idx]
                w = 1/(err + 1e-10)
            
            # 重建样条 (k=3, s=len/2 保证一定平滑度但不过拟合)
            # 注意：这里 s 不要设为 0，允许微小的平滑以抵抗噪音
            spl = UnivariateSpline(t, u4, w=w, k=3, s=len(t)*0.5)
            splines[L] = spl
            valid_L.append(L)
            
        except Exception as e:
            print(f"❌ 读取 L={L} 失败: {e}")

    # 2. 计算所有相邻对的交点
    crossings = []
    print(f"\n📋 各尺寸对交点详情:")
    print(f"-"*40)
    print(f"{'Pairs (L1 vs L2)':<20} | {'Tc Estimate':<15}")
    print(f"-"*40)

    for i in range(len(valid_L)-1):
        L1 = valid_L[i]
        L2 = valid_L[i+1] # 或者两两组合，这里取相邻对最有代表性
        
        def diff_func(x):
            return splines[L1](x) - splines[L2](x)
        
        try:
            # 在 2.2 到 2.35 之间搜寻根
            root = brentq(diff_func, 2.20, 2.35)
            crossings.append(root)
            print(f"L={L1:<3} vs L={L2:<3}      | {root:.6f}")
        except:
            print(f"L={L1:<3} vs L={L2:<3}      | 未找到交点 (No Crossing)")

    # 3. 统计分析
    if not crossings:
        print("\n❌ 无法确定 Tc：没有找到任何有效交点。")
        return

    crossings = np.array(crossings)
    
    # 策略 A: 全量统计
    mean_all = np.mean(crossings)
    std_all  = np.std(crossings)
    
    # 策略 B: 剔除小尺寸 (L=16) 的优化统计
    # 小尺寸通常受有限尺寸效应 (Finite Size Scaling corrections) 影响大，偏离真实值
    # 如果交点数超过 2 个，建议剔除第一个（含 L=16 的那个）
    if len(crossings) > 2:
        optimized_crossings = crossings[1:] 
        mean_opt = np.mean(optimized_crossings)
        std_opt = np.std(optimized_crossings)
        note = "(已剔除 L=16 相关项以提高精度)"
    else:
        mean_opt = mean_all
        std_opt = std_all
        note = "(数据较少，使用全部数据)"

    print(f"\n{'-'*60}")
    print(f"🏆 最终结果 (Final Result) {note}")
    print(f"{'-'*60}")
    print(f"平均临界温度 Tc = {mean_opt:.5f}")
    print(f"统计误差范围 ±  = {std_opt:.5f}")
    print(f"置信区间 (2σ)   = [{mean_opt - 2*std_opt:.5f}, {mean_opt + 2*std_opt:.5f}]")
    
    # 与 Onsager 理论值对比
    onsager_tc = 2.269185
    diff = abs(mean_opt - onsager_tc)
    print(f"\n理论值偏差: {diff:.5f} ({diff/onsager_tc*100:.2f}%)")
    
    # 保存结果到txt
    with open(os.path.join(input_folder, 'Tc_Final_Calculation.txt'), 'w') as f:
        f.write(f"Tc_Mean = {mean_opt:.6f}\n")
        f.write(f"Tc_Std = {std_opt:.6f}\n")
        f.write(f"Raw_Crossings = {crossings.tolist()}\n")

if __name__ == "__main__":
    calculate_precise_tc_statistics()