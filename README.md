# RL-Beginner：强化学习任务驱动入门

本仓库面向已经掌握 Python、PyTorch 和基础深度学习的学习者，按照“原理与公式 → 手写实现 → 实验观察 → 自检复盘”的路径，完成从多臂老虎机到现代策略优化算法的强化学习入门。

当前阶段完成并整理到 **Task 7**。每个任务都包含源码、教学 Notebook、实验 Notebook、学习笔记和与算法对应的独立自检脚本。

## 学习路线

| 任务 | 主题 | 核心问题 | 主要实现 |
|---|---|---|---|
| Task 1 | 多臂老虎机 | 如何平衡探索与利用？ | Greedy、ε-greedy、UCB、Beta-Bernoulli Thompson Sampling |
| Task 2 | MDP 与动态规划 | 已知环境模型时如何求解价值和策略？ | GridWorld、Policy Evaluation、Policy Iteration、Value Iteration |
| Task 3 | 表格型强化学习 | 没有环境模型时如何从经验学习？ | Monte Carlo、TD(0)、SARSA、Q-Learning |
| Task 4 | DQN | 如何用神经网络近似动作价值函数？ | Q Network、Replay Buffer、TD 更新、Target Network |
| Task 5 | Policy Gradient | 如何直接优化随机策略？ | Return、REINFORCE、Baseline、策略梯度更新 |
| Task 6 | Actor-Critic | 如何降低策略梯度的方差并提高样本效率？ | Actor-Critic、n-step Actor-Critic、TD residual、GAE |
| Task 7 | TRPO / PPO | 如何在策略更新时控制步长并提升稳定性？ | Rollout Buffer、GAE、PPO clipping、TRPO trust region |

## Task 1–7 交付内容

| 任务 | 源码目录 | 学习 Notebook | 实验 Notebook | 自检 |
|---|---|---|---|---|
| Task 1 | [task-01-bandit/src](task-01-bandit/src/) | [01-learning.ipynb](task-01-bandit/notebooks/01-learning.ipynb) | [02-experiments.ipynb](task-01-bandit/notebooks/02-experiments.ipynb) | [eval/run.py](task-01-bandit/eval/run.py) |
| Task 2 | [task-02-mdp-dp/src](task-02-mdp-dp/src/) | [01-learning.ipynb](task-02-mdp-dp/notebooks/01-learning.ipynb) | [02-experiments.ipynb](task-02-mdp-dp/notebooks/02-experiments.ipynb) | [eval/run.py](task-02-mdp-dp/eval/run.py) |
| Task 3 | [task-03-tabular-rl/src](task-03-tabular-rl/src/) | [01-learning.ipynb](task-03-tabular-rl/notebooks/01-learning.ipynb) | [02-experiments.ipynb](task-03-tabular-rl/notebooks/02-experiments.ipynb) | [eval/run.py](task-03-tabular-rl/eval/run.py) |
| Task 4 | [task-04-dqn/src](task-04-dqn/src/) | [01-learning.ipynb](task-04-dqn/notebooks/01-learning.ipynb) | [02-experiments.ipynb](task-04-dqn/notebooks/02-experiments.ipynb) | [eval/run.py](task-04-dqn/eval/run.py) |
| Task 5 | [task-05-policy-gradient/src](task-05-policy-gradient/src/) | [01-learning.ipynb](task-05-policy-gradient/notebooks/01-learning.ipynb) | [02-experiments.ipynb](task-05-policy-gradient/notebooks/02-experiments.ipynb) | [eval/run.py](task-05-policy-gradient/eval/run.py) |
| Task 6 | [task-06-actor-critic/src](task-06-actor-critic/src/) | [01-learning.ipynb](task-06-actor-critic/notebooks/01-learning.ipynb) | [02-experiments.ipynb](task-06-actor-critic/notebooks/02-experiments.ipynb) | [eval/run.py](task-06-actor-critic/eval/run.py) |
| Task 7 | [task-07-trpo_ppo/src](task-07-trpo_ppo/src/) | [01-learning.ipynb](task-07-trpo_ppo/notebooks/01-learning.ipynb) | [02-experiments.ipynb](task-07-trpo_ppo/notebooks/02-experiments.ipynb) | [eval/run.py](task-07-trpo_ppo/eval/run.py) |

## 每个任务的学习重点

### Task 1：多臂老虎机

从无状态的 Bernoulli bandit 开始，理解真实价值、采样奖励、增量均值估计和 pseudo-regret。实验比较 Greedy、ε-greedy、UCB 与 Thompson Sampling 在探索行为、平均奖励和最优动作率上的差异。

### Task 2：MDP 与动态规划

在已知转移概率和奖励的 GridWorld 中实现 Bellman backup。先进行固定策略评估，再通过 Policy Iteration 和 Value Iteration 求解最优价值函数与策略，观察两种动态规划方法的收敛过程。

### Task 3：表格型强化学习

从模型无关的角度学习状态价值和动作价值估计。Notebook 对比 Monte Carlo 的完整回报、TD(0) 的 bootstrap、SARSA 的 on-policy 更新和 Q-Learning 的 off-policy 更新。

### Task 4：DQN

将表格 Q-Learning 扩展到神经网络，理解经验回放、目标网络和 TD target 对训练稳定性的作用。实验围绕 Q 网络输出、Replay Buffer 采样和 CartPole 风格控制任务展开。

### Task 5：Policy Gradient

直接对策略分布进行优化。先理解折扣回报和 log-probability，再实现 REINFORCE，并加入 baseline 和回报归一化以观察方差变化。原有多个学习 Notebook 已整理为统一的 `01-learning.ipynb` 和 `02-experiments.ipynb` 入口，同时保留原始材料。

### Task 6：Actor-Critic

用 Critic 估计价值并为 Actor 提供低方差学习信号。内容覆盖一步 TD、n-step return、terminal target 和 GAE。异常命名的原始 Notebook 已补充规范的 `01-learning.ipynb` 入口，实验统一放在 `02-experiments.ipynb`。

### Task 7：TRPO / PPO

在策略梯度基础上加入受约束的策略更新。Notebook 和源码覆盖 rollout 收集、GAE、PPO clipped objective、approximate KL、clip fraction，以及 TRPO 的 Fisher-vector product、共轭梯度和 line search。目录统一为 `task-07-trpo_ppo`。

## 统一目录约定

每个已完成任务遵循以下结构：

```text
task-XX-*/
├── README.md
├── src/                  # 算法实现
├── notebooks/
│   ├── 01-learning.ipynb
│   └── 02-experiments.ipynb
├── notes/                # 推导、实验记录与复盘
└── eval/
    └── run.py            # 当前任务专属的最小自检
```

Notebook 负责解释、可视化和实验调用，核心算法保留在 `src/`。公式使用 Markdown 的 `$$ ... $$` 块，确保 Jupyter 和 GitHub 页面都能正常渲染。

## 运行环境

推荐使用 Python 3.10 及以上的 Conda 环境：

```bash
conda activate llm-agent
pip install -r requirements.txt
```

不需要为了本仓库下载大模型；Task 1–7 的实验均以小型环境、表格数据或轻量神经网络为主。

## 运行自检

从仓库根目录运行全部 Task 1–7 的任务专属检查：

```bash
python eval/run.py
```

只检查某一个任务：

```bash
python eval/run.py --task 1
python eval/run.py --task 7
```

根目录脚本只负责调度；真正的检查逻辑位于每个任务自己的 `eval/run.py`，并根据任务算法验证不同的数值契约。例如：

- Task 1 检查 sample-average 更新和 Thompson Sampling 的二值奖励约束；
- Task 2 检查 GridWorld 转移、策略评估以及 PI/VI 的价值一致性；
- Task 3 检查 MC、TD、SARSA 和 Q-Learning 的更新结果；
- Task 4 检查 Q 网络、经验回放和 DQN TD 更新；
- Task 5 检查折扣回报、策略分布和 REINFORCE 更新；
- Task 6 检查一步 / n-step Actor-Critic 与 terminal target；
- Task 7 检查 GAE、rollout buffer、PPO 诊断量和 TRPO 构造。

自检用于快速发现接口和核心数值错误，不替代完整训练实验。实验结果应记录随机种子、环境、超参数、曲线和失败现象，并写入对应 `notes/`。

## 相关文档

- [强化学习速查表](docs/rl-cheatsheet.md)
- [数学预备知识](docs/math-basics.md)
- [术语表](docs/glossary.md)

当前 README 和本阶段交付范围截至 **Task 7**；后续任务将在单独整理后再加入路线图。

## 许可证

[MIT License](LICENSE)。
