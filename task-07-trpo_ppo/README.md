# 任务七：TRPO 与 PPO

## 目标

在 Actor-Critic 基础上限制策略更新幅度，理解 trust region、TRPO 和 PPO clipped objective。

$$
r_t(\theta)=\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\theta_{old}}(A_t\mid S_t)}.
$$

$$
L^{CLIP}=\mathbb{E}_t[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)].
$$

## 源码契约

目录已统一为 `task-07-trpo_ppo`：`network.py` 定义 Actor/Critic，`buffer.py` 保存 on-policy rollout，`advantage.py` 计算 GAE，`ppo.py` 实现多 epoch mini-batch PPO，`trpo.py` 提供 trust-region 对照。`dones` 应传真实 terminal mask；Gymnasium 的 `truncated` 要重置环境但仍允许 value bootstrap。

## Notebook 与实验

- `01-learning.ipynb`：策略分布、采样、log-prob、Actor-Critic 基础。
- `02-experiments.ipynb`：rollout、GAE、clip objective、PPO 训练、clip epsilon/lambda 与 baseline 对照。

固定 rollout budget 和 seed，记录 episode return、entropy、value loss、clip fraction、approximate KL，并解释 ratio 超出区间时 clip 如何保护更新。

```bash
python eval/run.py --task 7
```

实验记录写入 `notes/task07_trpo_ppo.md`。
