import numpy as np

class IsingSimulation:
    def __init__(self, L=50, T=2.0, J=1.0, dynamics="metropolis"):
        """
        L         : 线性格点数
        T         : 温度
        J         : 相互作用常数
        dynamics  : "metropolis" 或 "glauber"，指定动力学模型
        """
        self.L = L  # 格点尺寸
        self.N = L * L  # 总自旋数
        self.T = T  # 温度
        self.J = J  # 相互作用强度
        self.beta = 1.0 / T  # 逆温度
        self.dynamics = dynamics  # 动力学类型
        self.config = np.random.choice([1, -1], size=(L, L))  # 初始自旋配置

        # 初始化每步的磁化强度和能量采样
        self.m_samples = []
        self.e_samples = []

        # 初始化总能量与总磁化
        self.energy = self._compute_total_energy()
        self.magnetization = np.sum(self.config)
    def initialize_spins(self, mode='random'):
        """ 初始化自旋配置 """
        if mode == 'random':
            self.config = np.random.choice([1, -1], size=(self.L, self.L))
        elif mode == 'aligned':
            self.config = np.ones((self.L, self.L), dtype=int) # 全正 1
        
        # 重置能量和磁化
        self.energy = self._compute_total_energy()
        self.magnetization = np.sum(self.config)
        self.m_samples = [] 
        self.e_samples = []

    def _get_nb_sum(self, i, j):
        """ 计算格点 (i, j) 的邻居自旋总和（周期性边界条件） """
        L = self.L
        return (self.config[(i + 1) % L, j] + self.config[(i - 1) % L, j] +
                self.config[i, (j + 1) % L] + self.config[i, (j - 1) % L])

    def _compute_total_energy(self):
        """ 计算整个系统的总能量（每对相互作用计算一次） """
        energy = 0.0
        L = self.L
        for i in range(L):
            for j in range(L):
                s = self.config[i, j]
                nb = self._get_nb_sum(i, j)
                energy += -self.J * s * nb
        return energy / 2.0  # 每对相互作用被计算了两次

    def _accept_metropolis(self, dE):
        """ Metropolis 算法的接受概率 """
        return dE <= 0.0 or np.random.rand() < np.exp(-self.beta * dE)

    def _accept_glauber(self, dE):
        """ Glauber 算法的接受概率 """
        return np.random.rand() < 1.0 / (1.0 + np.exp(self.beta * dE))

    def step(self):
        """
        单次 Monte Carlo 步骤（更新一个自旋）
        """
        L = self.L

        for _ in range(self.N):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)

            s = self.config[i, j]
            nb = self._get_nb_sum(i, j)
            dE = 2.0 * self.J * s * nb

            if self.dynamics == "metropolis":
                accept = self._accept_metropolis(dE)
            elif self.dynamics == "glauber":
                accept = self._accept_glauber(dE)
            else:
                raise ValueError("Unknown dynamics type")

            if accept:
                self.config[i, j] *= -1
                self.energy += dE
                self.magnetization += -2 * s

    @property
    def energy_density(self):
        """ 每个格点的平均能量 """
        return self.energy / self.N

    @property
    def magnetization_density(self):
        """ 每个格点的平均磁化强度 """
        return self.magnetization / self.N

    def run(self, steps, burn_in=1000):
        """
        运行模拟，计算系统在给定温度下的磁化强度和能量。
        保存每步的磁化强度和能量样本，并返回平均值。
        """
        # 预热阶段
        for _ in range(burn_in):
            self.step()

        # 采样阶段
        m_sum = 0.0
        e_sum = 0.0
        for _ in range(steps):
            self.step()
            m_sample = self.magnetization_density
            e_sample = self.energy_density

            # 保存每步的数据
            self.m_samples.append(m_sample)
            self.e_samples.append(e_sample)

            m_sum += m_sample
            e_sum += e_sample

        # 计算最终的平均值
        m_avg = m_sum / steps
        e_avg = e_sum / steps

        return m_avg, e_avg
