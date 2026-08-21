# 任务二：MDP、Bellman 方程与动态规划

## 目标

在 `GridWorld` 中把 Bandit 的一次决策扩展为序列决策，理解 policy evaluation、policy iteration 和 value iteration。Bellman expectation backup 为：

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma(1-d)V^\pi(s')].
$$

其中 `d` 表示这次 transition 是否真正终止；终止后不能 bootstrap。

## 学习与实验要求

1. 画出 GridWorld 状态、动作、障碍、起点和终点约定。
2. 对一个状态手算一次 action value，再运行策略评估。
3. 比较 policy iteration 与 value iteration 的收敛轮数、价值和策略。
4. 扫描 `gamma`，解释折扣因子对起点价值和路径偏好的影响。

## 当前源码契约

| 文件 | 作用 |
|---|---|
| `src/gridworld.py` | GridWorld 状态转移、奖励和终止语义 |
| `src/policy_evaluation.py` | 均匀策略、同步策略评估 |
| `src/policy_iteration.py` | 策略评估与贪心改进 |
| `src/value_iteration.py` | Bellman optimality backup |
| `src/visualization.py` | 价值图与策略箭头 |

## Notebook 架构

- `notebooks/01-learning.ipynb`：从环境接口到 Bellman backup、PI/VI。
- `notebooks/02-experiments.ipynb`：收敛轨迹、算法对照和 gamma 消融。

## 运行与提交

```bash
cd /workspace/YiFan/llm_agent/repos/RL-beginner
python eval/run.py --task 2
```

将推导和结果写入 `notes/task-02-notes.md`，不要只提交图片。
