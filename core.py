import numpy as np
from numba import njit


@njit
def init_lattice(L):
    """初始化晶格, 默认由随机态开始"""
    config = np.random.choice(np.array([-1, 1]), (L, L)) 
    # np.array() 非必要, 但最好加
    return config


@njit
def Metropolis_kernel(config, L, steps, beta):
    """核心代码块, 自动计算 E 和 M """
    # 初始化
    E_list = np.zeros(steps)
    M_list = np.zeros(steps)

    # 查找法先计算列表
    prob_table = np.array([1.0, np.exp(-4 * beta), np.exp(-8 * beta)])

    # 先计算初态 E 和 M, 后续才能进行累加
    E = 0.0   # Numba中浮点数必须写成浮点的样子
    M = 0.0

    for i in range(L):
        for j in range(L):
            s = config[i, j]
            M += s
            nb = config[(i+1)%L, j] + config[(i-1)%L, j] + config[i, (j+1)%L] + config[i, (j-1)%L]
            E += -s * nb / 2.0

    # 主循环
    for n in range(steps):
        for _ in range(L ** 2):
            # 随机选某个格点
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)

            # 计算dE
            s = config[i, j]
            nb = config[(i+1)%L, j] + config[(i-1)%L, j] + config[i, (j+1)%L] + config[i, (j-1)%L]
            dE = 2 * s * nb

            # Metropolis 能量判据
            if dE < 0 or np.random.rand() < prob_table[dE // 4]:
                    E += dE
                    M += -2 * s
                    s *= -1
                    config[i, j] = s
        E_list[n] = E
        M_list[n] = M
    return (E_list, M_list)


@njit
def Glauber_kernel(config, L, steps, beta):
    """核心代码块, 自动计算 E 和 M """
    # 初始化
    E_list = np.zeros(steps)
    M_list = np.zeros(steps)

    # 查找法先计算列表
    prob_table = np.array([np.exp(8*beta) / (1+np.exp(8*beta)), np.exp(4*beta)/(1+np.exp(4*beta)),
                    0.5, np.exp(-4*beta) / (1+np.exp(-4*beta)), np.exp(-8*beta)/(1+np.exp(-8*beta))])

    # 先计算初态 E 和 M, 后续才能进行累加
    E = 0.0   # Numba中浮点数必须写成浮点的样子
    M = 0.0

    for i in range(L):
        for j in range(L):
            s = config[i, j]
            M += s
            nb = config[(i+1)%L, j] + config[(i-1)%L, j] + config[i, (j+1)%L] + config[i, (j-1)%L]
            E += -s * nb / 2.0

    # 主循环
    for n in range(steps):
        for _ in range(L ** 2):
            # 随机选某个格点
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)

            # 计算dE
            s = config[i, j]
            nb = config[(i+1)%L, j] + config[(i-1)%L, j] + config[i, (j+1)%L] + config[i, (j-1)%L]
            dE = 2 * s * nb

            # Glauber 能量判据
            if np.random.rand() < prob_table[dE // 4 + 2]:
                    E += dE
                    M += -2 * s
                    s *= -1
                    config[i, j] = s
        E_list[n] = E
        M_list[n] = M
    return (E_list, M_list)

