import numpy as np


def cal_M_avg(M_list, L):
    """
    M_list 应为一个一维数组
    自动求该数组取绝对值后单个粒子 M 的平均值
    """
    M_abs = np.abs(M_list)
    M_avg = np.mean(M_abs) / L ** 2
    return M_avg


def cal_E_avg(E_list, L):
    """
    E_list 应为一个一维数组
    自动求该数组单个粒子 E 的平均值
    """
    E_avg = np.mean(E_list) / L ** 2
    return E_avg


def cal_Cv(E_list, L, T):
    """
    E_list 应为一个一维数组
    输出比热容Cv
    """
    beta = 1 / T
    E_var = np.var(E_list)
    Cv = (E_var * beta ** 2) / L ** 2
    return Cv


def cal_Chi(M_list, L, T):
    """
    M_list 应为一个一维数组
    输出磁化率Chi
    """
    beta = 1 / T
    M_var = np.var(np.abs(M_list))
    Chi = (M_var * beta) / L ** 2
    return Chi


def cal_U4(M_list):
    """
    M_list 应为一维数组
    输出 Binder Cumulant U4
    """
    m2 = np.mean(M_list ** 2)
    m4 = np.mean(M_list ** 4)
    if m2 < 1e-15:
        return 0.0
    U4 = 1 - (m4 / (3*m2**2))
    return U4