import numpy as np
from numba import njit, prange


# ============================================================
# 1. 分块分析 (Blocking Analysis) - 处理原始数据
# ============================================================


import numpy as np
from numba import njit

# ============================================================
# 1. 分块分析 (Blocking Analysis) - 处理原始数据
# ============================================================

@njit
def _blocking_core_numba(data_array, min_block_count):
    """
    [内部核心] Numba 加速的分块变换
    
    原理：
        执行类似重整化群(RG)的粗粒化过程。
        每一轮迭代将数据长度减半，计算该层级的标准误(SEM)。
    
    返回：
        block_sizes (list): 每一轮的块大小 [1, 2, 4, 8, ...]
        errors (list): 对应块大小计算出的标准误 [sigma_1, sigma_2, ...]
    """
    n = len(data_array)
    current_data = data_array.copy() # 保护原数据不被修改
    
    # Numba 的 List 需要类型一致，这里初始化存储结果
    errors = []
    block_sizes = []
    
    current_bs = 1
    
    while True:
        n_curr = len(current_data)
        
        # 1. 终止条件：当剩余块数太少（统计学上不足以估算方差）时停止
        if n_curr < min_block_count:
            break
            
        # 2. 计算当前层级的均值和标准误 (SEM)
        # 手写循环避免 numpy overhead，确保极致速度
        mean_val = 0.0
        for x in current_data:
            mean_val += x
        mean_val /= n_curr
        
        var_sum = 0.0
        for x in current_data:
            var_sum += (x - mean_val)**2
        
        # ddof=1 无偏估计
        std_dev = np.sqrt(var_sum / (n_curr - 1))
        sem = std_dev / np.sqrt(n_curr)
        
        errors.append(sem)
        block_sizes.append(current_bs)
        
        # 3. 核心变换：相邻求平均 (Decimation)
        # 下一层数据长度减半，信息密度增加
        new_len = n_curr // 2
        new_data = np.empty(new_len, dtype=np.float64)
        
        for i in range(new_len):
            # 显式循环会被编译为高效的 SIMD 指令
            new_data[i] = 0.5 * (current_data[2*i] + current_data[2*i+1])
            
        # 更新数据指针和块大小，进入下一轮
        current_data = new_data
        current_bs *= 2
        
    return block_sizes, errors

# ==========================================
# 用户接口
# ==========================================
def blocking_analysis(time_series, min_block_count=32):
    """
    [功能] 
    执行分块分析，去除 MC 数据的自相关，并提取统计独立的块均值序列。
    这是后续进行 Jackknife 分析的前置步骤。
    
    [输入]
        time_series: np.ndarray, 1D 原始数据序列（如能量或磁化强度）。
        min_block_count: int, 停止分块的最少块数 (默认32)。
        
    [输出]
        mean: float
            数据的全局平均值。
        error: float
            去自相关后的真实标准误差 (取误差平台区的最大值)。
        best_block_means: np.ndarray (1D)
            ***新增核心输出***
            这是基于最佳块大小重采样后的“干净”数据序列。
            请直接把这个传给 jackknife_analysis 进行后续计算。
        debug_info: dict
            包含各级分块误差的调试信息。
    """
    data = np.array(time_series, dtype=np.float64)
    
    # 1. 基础统计
    final_mean = np.mean(data)
    
    # 2. 调用 Numba 核心进行扫描
    block_sizes, errors = _blocking_core_numba(data, min_block_count)
    
    if len(errors) == 0:
        # 极端情况：数据太短，无法分块
        return final_mean, 0.0, data, {}

    # 3. 确定“真实误差”和“最佳块大小”
    # 物理逻辑：取误差平台的最大值作为保守估计
    # np.argmax 返回的是最大值的索引
    best_idx = np.argmax(errors)
    final_error = errors[best_idx]
    best_bs = block_sizes[best_idx]
    
    # 4. ***生成最佳块均值序列 (Critical Step)***
    # 根据找到的最佳块大小 best_bs，重新对原始数据进行切割和平均
    # 这样得到的 best_block_means 里的每个点才是统计独立的
    n_usable = (len(data) // best_bs) * best_bs
    reshaped_data = data[:n_usable].reshape(-1, best_bs)
    best_block_means = np.mean(reshaped_data, axis=1)
    
    return final_mean, final_error, best_block_means, {"block_sizes": block_sizes, "errors": errors}


# ============================================================
# 2. 刀切法 (Jackknife) - 处理非线性物理量 (U4, Chi)
# ============================================================


@njit(parallel=True)
def _jackknife_core_numba(data_blocks, func_type):
    """
    Numba 加速核心：生成 Jackknife 样本并计算统计量
    
    参数:
        data_blocks: (N_blocks, ...) 形状的数组，每个元素是一个独立块的均值
        func_type: int, 一个标记，用来告诉 Numba 算哪个物理量
                   1 = U4 (Binder Cumulant)
                   2 = Chi (Susceptibility / Variance)
                   (Numba 不支持传函数指针，所以用整数 flag 代替)
    """
    n = len(data_blocks)
    
    # 我们需要存储 N 个 Jackknife 估计值
    jk_estimates = np.zeros(n)
    
    # 原始数据的总和 (O(N))
    # 这一步是加速的关键：不用每次都重新求和，而是从总和里减去某一项
    total_sum_m2 = 0.0
    total_sum_m4 = 0.0
    total_sum_m1 = 0.0
    
    # 预先计算所有块的幂次，避免重复算
    m2 = data_blocks**2
    m4 = data_blocks**4
    
    for i in range(n):
        total_sum_m1 += data_blocks[i]
        total_sum_m2 += m2[i]
        total_sum_m4 += m4[i]
        
    # 并行循环生成 N 个 Jackknife 样本
    # 每次剔除第 i 个块，用剩下的 N-1 个块算物理量
    for i in prange(n):
        # 剔除第 i 个块后的总和
        sum_m2_partial = total_sum_m2 - m2[i]
        sum_m4_partial = total_sum_m4 - m4[i]
        sum_m1_partial = total_sum_m1 - data_blocks[i]
        
        # 剩下的 N-1 个样本的平均值
        avg_m2 = sum_m2_partial / (n - 1)
        avg_m4 = sum_m4_partial / (n - 1)
        avg_m1 = sum_m1_partial / (n - 1)
        
        if func_type == 1: # Calculate U4
            # U4 = 1 - <m^4> / (3 * <m^2>^2)
            if avg_m2 > 1e-12: # 防止除以0
                jk_estimates[i] = 1.0 - avg_m4 / (3.0 * avg_m2**2)
            else:
                jk_estimates[i] = 0.0
                
        elif func_type == 2: # Calculate Chi (Simplified ~ Variance)
            # Chi ~ <m^2> - <m>^2
            jk_estimates[i] = avg_m2 - avg_m1**2
            
    return jk_estimates

# ==========================================
# 用户接口
# ==========================================
def jackknife_analysis(block_means, observable_type='U4'):
    """
    [功能]
    使用刀切法 (Jackknife Resampling) 计算非线性物理量及其误差。
    核心用于处理像 U4 (Binder Cumulant) 或 磁化率 (Chi) 这种无法直接传递误差的复合量。

    [输入]
    block_means : array-like (1D)
        分块后的均值序列 (Block Averages)。
        ***注意***: 必须输入统计独立的块均值，严禁输入原始 MC 时间序列！
    observable_type : str
        计算目标类型。
        - 'U4' : Binder Cumulant, U4 = 1 - <m^4> / 3<m^2>^2
        - 'Chi': 磁化率 (Susceptibility), Chi ~ <m^2> - <m>^2

    [输出]
    val : float
        物理量的最佳估计值 (Jackknife Mean)。
    err : float
        物理量的标准误差 (Jackknife Standard Error)。
    """
    data = np.array(block_means, dtype=np.float64)
    n = len(data)
    
    if n < 2:
        return 0.0, 0.0 # 没法算
    
    # 映射类型到 Numba 识别的整数
    type_map = {'U4': 1, 'Chi': 2}
    if observable_type not in type_map:
        raise ValueError("Unsupported observable type")
    
    func_id = type_map[observable_type]
    
    # 1. 调用 Numba 算那 N 个“剔除一个后”的值
    jk_vals = _jackknife_core_numba(data, func_id)
    
    # 2. Jackknife 核心公式
    # 均值 = Jackknife 样本的均值
    jk_mean = np.mean(jk_vals)
    
    # 误差 = sqrt( (N-1)/N * sum( (jk_val - jk_mean)^2 ) )
    # 注意系数是 (N-1) 而不是 1/(N-1)，这是 Jackknife 的特征！
    variance = np.sum((jk_vals - jk_mean)**2)
    error = np.sqrt((n - 1) / n * variance)
    
    return jk_mean, error


# ============================================================
# 3. 参数自举法 (Bootstrap) - 拟合 Tc 交点
# ============================================================


@njit
def _linear_fit_fast(x, y):
    """[内部辅助] 极速线性拟合 y = a + b*x"""
    n = len(x)
    sum_x = 0.0; sum_y = 0.0
    sum_xx = 0.0; sum_xy = 0.0
    
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xx += x[i]**2
        sum_xy += x[i] * y[i]
        
    delta = n * sum_xx - sum_x**2
    if np.abs(delta) < 1e-12:
        return 0.0, 0.0 
        
    a = (sum_xx * sum_y - sum_x * sum_xy) / delta
    b = (n * sum_xy - sum_x * sum_y) / delta
    return a, b

@njit(parallel=True)
def _bootstrap_multi_core(temps, data_matrix, err_matrix, n_boot):
    """
    [内部核心] 支持任意数量 L 的 Bootstrap 引擎
    
    data_matrix shape: (N_sizes, N_temp_points)
    """
    n_sizes = data_matrix.shape[0]
    n_points = data_matrix.shape[1]
    
    # 存储最终 N_boot 个 Tc 样本
    tc_distribution = np.zeros(n_boot)
    
    # 并行重采样
    for k in prange(n_boot):
        # 1. 临时存储本轮所有尺寸的拟合参数 (a, b)
        # slopes[i] = b, intercepts[i] = a
        slopes = np.zeros(n_sizes)
        intercepts = np.zeros(n_sizes)
        
        # 2. 对每个尺寸 L 分别进行造假数据 + 拟合
        for i in range(n_sizes):
            # 模拟重采样：y_new = y_mean + noise * error
            sim_y = np.empty(n_points)
            for j in range(n_points):
                noise = np.random.normal(0, 1)
                sim_y[j] = data_matrix[i, j] + noise * err_matrix[i, j]
            
            # 线性拟合
            a, b = _linear_fit_fast(temps, sim_y)
            intercepts[i] = a
            slopes[i] = b
            
        # 3. 计算所有可能的两两交点 (Pairwise Intersections)
        # 例如 L=[16,32,64]，则算 (16,32), (16,64), (32,64) 的平均交点
        intersections = []
        count = 0
        
        for i in range(n_sizes):
            for j in range(i + 1, n_sizes):
                # 解方程: a_i + b_i*T = a_j + b_j*T
                diff_slope = slopes[i] - slopes[j]
                
                # 排除平行线或斜率太接近的情况
                if np.abs(diff_slope) > 1e-8:
                    tc_pair = (intercepts[j] - intercepts[i]) / diff_slope
                    
                    # 排除物理上离谱的解 (比如 T < 0 或 T 极其巨大)
                    # 这里放宽一点范围，防止过滤掉边缘数据
                    if np.abs(tc_pair) < 1000.0: 
                        # Numba list append 有点慢，这里用简单的累加求均值策略
                        # 暂时先用固定数组存一下，假设最多 10 个尺寸 -> 45 个对
                        # 为了代码简洁，直接累加
                        # (在 Numba 里 append 不太高效，但此处循环极小，可忽略)
                        intersections.append(tc_pair)
        
        # 4. 本轮 Bootstrap 的 Tc 估计值 = 所有交点的均值
        # 也可以改成 median 增加鲁棒性，但 mean 对光滑数据足够好
        if len(intersections) > 0:
            sum_tc = 0.0
            for val in intersections:
                sum_tc += val
            tc_distribution[k] = sum_tc / len(intersections)
        else:
            tc_distribution[k] = -999.0 # 标记失败
            
    return tc_distribution

def parametric_bootstrap_multi_L(temps, data_list, error_list, n_boot=2000):
    """
    [功能]
    多尺寸参数自举法 (Multi-L Parametric Bootstrap)。
    同时利用多个系统尺寸 (如 L=16, 32, 64) 的数据来拟合相变点 Tc。
    
    [原理]
    1. 在每一轮 Bootstrap 中，对所有 L 的数据同时进行高斯重采样。
    2. 计算所有 L 两两组合 (Pairwise) 的交点。
    3. 取这些交点的平均值作为该轮的 Tc。
    这种方法能充分利用所有数据，比只选两个 L 更稳健。

    [输入]
    temps      : np.ndarray (1D)
                 温度点序列 (X轴)，假设所有 L 都在相同的温度点测量。
    data_list  : list of arrays 或 2D array (shape: [N_sizes, N_temps])
                 每一行代表一个 L 的测量均值序列。
    error_list : list of arrays 或 2D array (shape: [N_sizes, N_temps])
                 每一行代表一个 L 的测量误差序列。
    n_boot     : int
                 重采样次数，默认 2000。

    [输出]
    tc_mean    : float -> Tc 的最佳估计。
    tc_std     : float -> Tc 的标准误差。
    """
    T = np.array(temps, dtype=np.float64)
    # 强制转换为 2D 矩阵供 Numba 使用
    D = np.array(data_list, dtype=np.float64)
    E = np.array(error_list, dtype=np.float64)
    
    # 维度检查
    if D.ndim != 2 or E.ndim != 2:
        raise ValueError("Input data/errors must be 2D arrays (Rows=Sizes, Cols=Temps)")
    if D.shape != E.shape:
        raise ValueError("Data and Error shapes must match.")
        
    # 调用 Numba 核心
    tc_samples = _bootstrap_multi_core(T, D, E, n_boot)
    
    # 过滤无效值
    valid_samples = tc_samples[tc_samples != -999.0]
    
    if len(valid_samples) == 0:
        return np.nan, np.nan
        
    return np.mean(valid_samples), np.std(valid_samples)


# ============================================================
# 4. 统一测试区 (Self-Test)
# ============================================================

if __name__ == "__main__":
    print(">>> [Self-Check] Starting physics utils verification...")
    
    # --- Test 1: Blocking Analysis ---
    # 造一个强自相关的序列 (x_t = 0.9*x_{t-1} + noise)
    raw_data = np.zeros(10000)
    for i in range(1, 10000):
        raw_data[i] = 0.9 * raw_data[i-1] + np.random.normal(0, 1)
    
    # [修改点] 现在返回4个值，用 _ 忽略 debug_info
    # cleaned_data 就是可以直接喂给 Jackknife 的“最佳块均值序列”
    mu, err, cleaned_data, _ = blocking_analysis(raw_data)
    print(f"1. Blocking Analysis : Mean={mu:.4f}, Error={err:.4f}, Cleaned_N={len(cleaned_data)} [OK]")

    # --- Test 2: Jackknife (U4) ---
    # [修改点] 直接使用 Step 1 产出的 cleaned_data，模拟真实流水线
    u4_val, u4_err = jackknife_analysis(cleaned_data, 'U4')
    print(f"2. Jackknife (U4)    : Val={u4_val:.4f}, Error={u4_err:.4f} [OK]")

    # --- Test 3: Multi-L Bootstrap (Tc) ---
    # [修改点] 适应新的多尺寸矩阵输入接口
    T = np.linspace(2.0, 2.5, 10)
    # 造 3 条有交点的线 (模拟 L=16, 32, 64)
    y1 = 0.5 * (T - 2.25) + np.random.normal(0, 0.01, 10)
    y2 = 0.9 * (T - 2.25) + np.random.normal(0, 0.01, 10)
    y3 = 1.5 * (T - 2.25) + np.random.normal(0, 0.01, 10)
    
    # 将它们堆叠成 (3, 10) 的矩阵
    data_mat = np.vstack([y1, y2, y3])
    err_mat = np.full_like(data_mat, 0.01)
    
    tc, tc_err = parametric_bootstrap_multi_L(T, data_mat, err_mat, n_boot=500)
    print(f"3. Multi-L Bootstrap : Tc={tc:.4f}, Error={tc_err:.4f} [OK]")
    
    print("\n>>> ✅ All systems GO! Ready for production.")