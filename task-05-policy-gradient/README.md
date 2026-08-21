# 任务五：Policy Gradient、REINFORCE 与 Baseline

## 目标

直接学习离散动作策略 `pi_theta(a|s)`，理解 likelihood-ratio 梯度、完整回报和 baseline 的方差控制。

$$
\nabla_\theta J(\theta)\approx\sum_tG_t\nabla_\theta\log\pi_\theta(A_t\mid S_t).
$$

加入状态 baseline 后：

$$
A_t=G_t-V_\phi(S_t).
$$

## 源码契约

`policy_network.py` 定义策略分布与动作采样；`reinforce.py` 提供回报计算和 vanilla 更新；`reinforce_baseline.py` 提供 value network、baseline 更新、训练和评估。

## Notebook 架构

本任务原来有四份 Notebook。现在统一提供：

- `notebooks/01-learning.ipynb`：合并 policy basics、REINFORCE、REINFORCE+baseline 三段学习内容。
- `notebooks/02-experiments.ipynb`：多 seed、最终窗口和达到阈值的对照实验。

原始分段 Notebook 保留作细粒度参考，统一入口才是提交路径。

## 实验要求

至少 5 个 seed，固定 CartPole 和训练预算，比较 vanilla 与 baseline 的均值、标准差、收敛速度和 loss；不要只比较单次最高分。

```bash
python eval/run.py --task 5
```

实验记录写入 `notes/task05_policy_gradient.md`。
