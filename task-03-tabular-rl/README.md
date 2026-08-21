# 任务三：表格型 RL：MC、TD、SARSA 与 Q-Learning

## 目标

从已知模型的 DP 转向 model-free 学习：智能体只能看到 `(S_t,A_t,R_{t+1},S_{t+1})`，通过采样估计价值和动作价值。

## 核心公式

完整回报：

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

TD(0)：

$$
V(S_t)\leftarrow V(S_t)+\alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)].
$$

SARSA 使用实际下一动作：

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)].
$$

Q-Learning 使用贪心目标：

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma\max_aQ(S_{t+1},a)-Q(S_t,A_t)].
$$

## 源码与 Notebook

`src/` 提供 `GridWorld`、`first_visit_mc_prediction`、`td0_prediction`、`sarsa` 和 `q_learning`。`01-learning.ipynb` 讲 prediction，`02-experiments.ipynb` 统一比较 DP/MC/TD 与 SARSA/Q-Learning。

## 实验要求

固定环境、策略、epsilon、alpha、gamma 和至少一个 seed；优先用 RMSE、平均 episode return、到达目标步数和最终策略解释差异。必须明确终止状态是否 bootstrap。

## 自检

```bash
python eval/run.py --task 3
```

实验记录写入 `notes/task-03-tabular-rl.md`。
