import numpy as np
import core
import utils.data_cal as cal
import matplotlib.pyplot as plt


# =================================================
#             参   数   配   置   区                
# =================================================



# 空间参数
L_list = np.array([16, 32, 64, 80])

# 时间参数
burn_in = 1000
steps = 2000

# 温度参数
T_start, T_end = 100000, 1000000 
T_points = 10
T_range = np.linspace(T_start, T_end, T_points)



# =================================================
#             实   验   主   循   环
# =================================================


# 初始化参数
# 预分配测量量
M_avg = np.zeros(T_points)
E_avg = np.zeros(T_points)
Chi = np.zeros(T_points)
Cv = np.zeros(T_points)

# 对L进行循环
for L in L_list:
    # 在温度循环之外初始化格点, 可以让每个温度的步数都累进, 更快到达平衡态
    config = core.init_lattice(L)

    # 对T进行循环
    for i, T in enumerate(T_range):
        beta = 1 / T
        # 先热化, 不记录数据
        core.Metropolis_kernel(config, L, burn_in, beta)
        
        # 开始测量
        E_list, M_list = core.Metropolis_kernel(config, L, steps, beta)

        # 计算统计量
        M_avg[i] = cal.cal_M_avg(M_list, L)
        E_avg[i] = cal.cal_E_avg(E_list, L)
        Chi[i] = cal.cal_Chi(M_list, L, T)
        Cv[i] = cal.cal_Cv(E_list, L, T)
        if i % 5 == 0:
            print(f"L = {L}, T = {T:.3}, 第{i}个温度点")

    np.savez(f"data/Ising_L{L}_steps{steps}.npz",
            M = M_avg,
            E = E_avg,
            Chi = Chi,
            Cv = Cv)
    plt.plot(T_range, M_avg, 'o-')
    plt.show()







