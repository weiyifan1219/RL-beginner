# 任务六：Actor-Critic、n-step 与 GAE

## 目标

让 Actor 学策略、Critic 学状态价值，用 TD residual 降低 REINFORCE 的方差。

$$
\delta_t=R_{t+1}+\gamma(1-d_t)V(S_{t+1})-V(S_t).
$$

GAE 将多个时间尺度的 residual 衰减组合：

$$
\hat A_t^{GAE(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
$$

## 源码契约

`src/networks.py` 提供 Actor/Critic；`actor_critic.py` 是 one-step agent；`n_step_actor_critic.py` 是 n-step agent。终止和时间截断要区分，只有真正终止才清零 bootstrap。

## Notebook 架构

规范入口为 `notebooks/01-learning.ipynb` 与 `notebooks/02-experiments.ipynb`。前者由原来的 `actor_critic.ipynb.ipynb` 整理而来，后者用于 one-step/n-step、lambda 和训练曲线对照；原文件保留。

## 实验要求

固定环境和 rollout budget，比较 one-step 与 n-step；若运行 GAE 扩展，至少比较 lambda=0、0.95、1，记录 return、actor loss、critic loss、entropy 和达到阈值的 episode。

```bash
python eval/run.py --task 6
```

实验记录写入 `notes/task06_actor_critic_notes.md`。
