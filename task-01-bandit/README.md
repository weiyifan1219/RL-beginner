# 任务一：多臂老虎机与探索

## 目标

在没有状态转移的 Bernoulli 寻宝环境中，理解探索/利用并运行 Greedy、ε-greedy、UCB 和 Beta-Bernoulli Thompson Sampling。真实价值为：

$$
q_*(a)=\mathbb{E}[R_t\mid A_t=a].
$$

## 学习与实验要求

1. 解释 `R_t`、真实价值 `q_*(a)` 与估计值 `Q_t(a)` 的区别。
2. 推导并观察 sample-average 更新：

$$
Q_{n+1}(a)=Q_n(a)+\frac{1}{N_n(a)}[R_n-Q_n(a)].
$$

3. 在同一组概率上比较四种策略，至少展示估计值、动作访问次数、平均奖励和最优动作率。
4. 说明 Thompson Sampling 只接受 0/1 奖励；Gaussian 与非平稳扩展属于后续练习。

## 当前源码契约

| 文件 | 作用 |
|---|---|
| `src/treasure_bandit.py` | `TreasureHuntBandit.step()`、`optimal_action`、`optimal_value` |
| `src/greedy.py` | sample-average Greedy |
| `src/epsilon_greedy.py` | ε-greedy |
| `src/ucb.py` | UCB，先访问未尝试动作 |
| `src/thompson_sampling.py` | Beta-Bernoulli 后验采样 |

## Notebook 架构

- `notebooks/01-learning.ipynb`：环境、更新、边界检查和最小策略对比。
- `notebooks/02-experiments.ipynb`：单轨迹与多 seed 实验。

Notebook 只调用 `src/`，不复制算法。

## 运行与提交

```bash
cd /workspace/YiFan/llm_agent/repos/RL-beginner
python eval/run.py --task 1
```

提交 `README.md`、`notes/task-01-notes.md`、Notebook 和实验输出说明；不要提交模型或生成日志。
