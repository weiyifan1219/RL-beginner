# 任务四：DQN 与稳定的价值学习

## 目标

将表格 Q-Learning 推进到 CartPole 上的函数逼近。DQN 的 TD target 为：

$$
y_t=r_t+\gamma(1-d_t)\max_{a'}Q_{\theta^-}(s_{t+1},a').
$$

其中 `theta-` 是 target network 参数。

## 源码契约

| 文件 | 作用 |
|---|---|
| `src/q_network.py` | 状态到各动作 Q 值的 MLP |
| `src/replay_buffer.py` | transition 存储与随机 batch 采样 |
| `src/dqn_agent.py` | epsilon-greedy、TD loss、target 同步 |

## 学习与实验要求

1. 先检查网络输出形状和 replay buffer 样本形状。
2. 在 CartPole 训练并分离训练/评估环境。
3. 记录 epsilon、episode return、loss 和评估回报。
4. 做一个单变量消融：去掉 replay 或 target network，解释稳定性变化。

`01-learning.ipynb` 负责接口和 TD target，`02-experiments.ipynb` 负责训练、评估和曲线。

```bash
python eval/run.py --task 4
```

实验记录写入 `notes/task04_dqn.md`。
