# 2D Ising Model Simulation Framework

![Status](https://img.shields.io/badge/Status-Research_Grade-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Acceleration](https://img.shields.io/badge/Numba-JIT_Accelerated-orange)

基于 Python (Numba) + Monte Carlo 的二维 Ising 模型高性能仿真与相变分析框架。本项目通过核心误差分析体系（Blocking Analysis, Jackknife, Parametric Bootstrap），精确求解临界温度 Tc 与临界指数。

## Core Features

* High-Performance Core: 核心 Metropolis 算法采用 numba JIT 编译，模拟速度接近 C++。
* Rigorous Error Analysis: 内置完整的统计物理误差处理流水线，自动去除马尔科夫链自相关效应。
* Publication-Ready Plotting: 包含针对 U4 交叉点、临界指数拟合的自动化绘图脚本。
* Structured Workflow: 严格分离 Core (计算)、Utils (统计)、Plots (绘图) 与 Data (存储)。

## Project Structure

```text
ISING_PROJECT/
├── core.py                 # Numba 加速的 MC 模拟引擎
├── main.py                 # 仿真主程序入口
├── utils/                  # 物理与统计工具箱
│   ├── data_cal.py         # 物理量计算
│   └── data_err.py         # 误差分析核心 (Blocking, Jackknife, Bootstrap)
├── plots/                  # 绘图脚本库
├── data/                   # 存放原始 MC 序列
├── figures/                # 自动生成的图表产出
│   ├── publication/        # 用于论文的矢量图
│   └── debug/              # 调试中间结果
└── archive/                # 历史归档