# 任务一：多臂老虎机与探索

> 本任务的核心不是“背三种策略”，而是建立强化学习最先遇到的矛盾：有限交互预算下，应该利用当前看来最好的动作，还是探索不确定的动作？

## 一句话目标

从零实现 Bernoulli/Gaussian 多臂老虎机、增量价值估计、ε-greedy、UCB 与 Beta-Bernoulli Thompson Sampling，并用平均奖励、最优动作率和累计 pseudo-regret 解释它们的行为。

## 学完后你应该能回答

1. 为什么只选当前经验均值最大的动作可能永远找不到最优臂？
2. 为什么样本平均适合平稳问题，而固定步长更适合非平稳问题？
3. ε-greedy、UCB、Thompson Sampling 分别如何表达“不确定性”？
4. 为什么 pseudo-regret 不应使用本次带噪声的 sampled reward？
5. 为什么本任务的 Beta Thompson Sampling 不能直接用于 Gaussian 奖励？

## 1. 问题定义

一个 `K` 臂老虎机只有动作，没有状态转移。第 `t` 步选择动作 `A_t=a` 后得到随机奖励 `R_t`。动作真实价值为：

```text
q*(a) = E[R_t | A_t = a]
```

学习者只能看到采样奖励，不能直接读取 `q*(a)`。实验代码知道真实值，只为了计算指标。

本任务提供两种环境：

| 环境 | 奖励分布 | 参数 | 适合观察 |
|---|---|---|---|
| `BernoulliBandit` | `R ~ Bernoulli(p_a)` | 每臂成功概率 `p_a` | 三种策略公平对比；Beta 共轭后验 |
| `GaussianBandit` | `R ~ N(q*(a), σ²)` | 每臂均值和共享标准差 | 连续噪声、乐观初值 |
| `NonStationaryGaussianBandit` | 均值随时间随机游走 | 初始均值、奖励噪声、漂移噪声 | 样本平均与固定步长对比 |

## 2. 增量价值估计

样本平均无需保存完整历史：

```text
Q_{n+1}(a) = Q_n(a) + 1/N_n(a) * [R_n - Q_n(a)]
```

方括号中的 `R_n - Q_n(a)` 是 estimation error。对非平稳环境，久远样本不应一直拥有同样权重，因此改成固定步长：

```text
Q_{n+1}(a) = Q_n(a) + α * [R_n - Q_n(a)],  0 < α <= 1
```

对应代码是 `src.agents.incremental_update`。`step_size=None` 使用样本平均；提供 `step_size` 则使用固定步长。

## 3. 三种探索策略

### 3.1 ε-greedy

```text
概率 1-ε：在 Q(a) 最大的动作中随机选一个
概率 ε：在所有动作中均匀随机选一个
```

它简单可靠，但探索是“盲目的”：已经证明确实很差的动作仍会以相同概率被探索。

### 3.2 Upper Confidence Bound（UCB）

```text
A_t = argmax_a [ Q_t(a) + c * sqrt(log(t) / N_t(a)) ]
```

第一项是利用，第二项是 uncertainty bonus。某动作访问越少，bonus 越大；总时间越长，长期未访问动作重新获得关注。实现会先确保每个动作至少试一次，避免除零。

### 3.3 Thompson Sampling

Bernoulli 成功概率使用 Beta 先验：

```text
p_a ~ Beta(alpha_a, beta_a)
成功：alpha_a <- alpha_a + 1
失败：beta_a  <- beta_a  + 1
```

每一步从各臂后验采样一个成功率，再选样本最大的动作。数据少时后验宽、自然多探索；数据多时后验收窄、自然多利用。

> 本仓库实现的是 **Beta-Bernoulli Thompson Sampling**，`update` 只接受 0/1。Gaussian 奖励需要 Gaussian likelihood 对应的另一套后验，不能把连续奖励硬塞进 Beta 更新。

## 4. 评价指标

| 指标 | 形状 | 含义 |
|---|---:|---|
| `rewards` | `(T,)` | 一次轨迹实际采样奖励，含噪声 |
| `optimal_actions` | `(T,)` | 每步是否选择当时的最优臂 |
| `instantaneous_regret` | `(T,)` | `max_a q*(a) - q*(A_t)` |
| `cumulative_regret` | `(T,)` | 即时 pseudo-regret 的累计和 |
| 聚合曲线 | `(T,)` | 对 `R` 次独立实验逐时间步求均值 |

若存在多个并列最优臂，选择其中任意一个都计为最优动作。`reward_standard_error`
由不同 run 估计；当 `n_runs=1` 时统计上无法估计，返回 `NaN` 而不是虚假的 0。

这里使用 pseudo-regret，而不是 `best_mean - sampled_reward`。后者会因奖励噪声变成负数，使“累计遗憾”下降，失去清晰含义。

## 5. 标准答案 API 与输入输出

### 环境

```python
from src.bandits import BernoulliBandit

env = BernoulliBandit([0.1, 0.5, 0.9], seed=42)
reward: float = env.pull(action=2)
means = env.expected_rewards       # float64, shape (3,)，返回副本
best: int = env.optimal_arm
gap: float = env.regret(action=0)  # 单步期望遗憾
```

### Agent

```python
from src.agents import EpsilonGreedyAgent

agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.1, step_size=None, seed=42)
action: int = agent.select_action()
agent.update(action, reward)
values = agent.estimates  # float64, shape (3,)
counts = agent.counts     # int64, shape (3,)
```

### 单次与多次实验

```python
from src.experiment import run_episode, run_experiment

episode = run_episode(env, agent, n_steps=100)
# episode.actions/rewards/... 的形状均为 (100,)

result = run_experiment(
    env_factory=lambda seed: BernoulliBandit([0.1, 0.5, 0.9], seed=seed),
    agent_factory=lambda seed: EpsilonGreedyAgent(3, epsilon=0.1, seed=seed),
    n_runs=200,
    n_steps=1000,
    seed=42,
)
# result.mean_reward/optimal_action_rate/... 的形状均为 (1000,)
```

环境和 agent 的随机数来自独立的 `SeedSequence` 子流。这样即使策略多抽一次随机数，也不会偷偷改变环境将产生的奖励序列。

## 6. 文件地图

| 文件 | 内容 |
|---|---|
| `src/bandits.py` | 三类 bandit 环境 |
| `src/agents.py` | 增量更新与三类策略 |
| `src/experiment.py` | 单轨迹、多 seed 聚合和绘图 |
| `run_experiment.py` | YAML 驱动的三策略实验 CLI |
| `configs/default.yaml` | 默认环境、预算和超参数 |
| `tests/` | 公式、边界、复现性、学习行为和 CLI 契约 |
| `eval/run.py` | 一键自检，结果写入 `eval/result.json` |
| `notebooks/task-01-bandit.ipynb` | 可直接运行的中文教学 Notebook |
| `notes/standard-answer.md` | 推导、设计决策和实验解读 |

## 7. 运行方式

在 3090 上：

```bash
cd /workspace/YiFan/llm_agent/repos/RL-beginner
conda activate llm-agent

# 快速自动检查
python task-01-bandit/eval/run.py

# 快速实验
python task-01-bandit/run_experiment.py --quick \
  --output-dir task-01-bandit/outputs/quick

# 默认 200 runs × 1000 steps
python task-01-bandit/run_experiment.py \
  --output-dir task-01-bandit/outputs/default

# Jupyter
cd task-01-bandit
python -m jupyter lab
```

Notebook 请选择 `python3` 内核。核心算法全部在 `.py` 文件中；Notebook 只导入和讲解，因此既可交互学习，也能被测试程序从干净 kernel 自动执行。

CLI 输出：

```text
outputs/<run>/
├── resolved_config.yaml
├── summary.json
├── curves.npz
└── learning_curves.png
```

## 8. 建议学习顺序

1. 不看标准答案，手算奖励 `[2, 4, 0]` 的样本平均更新。
2. 阅读测试，根据测试中的接口自己实现一个最小版本。
3. 运行 `pytest -q task-01-bandit/tests`，逐项修正。
4. 阅读 `src/` 标准答案，对比你在随机性、边界检查和实验聚合上的差异。
5. 顺序运行 Notebook，解释每张图，而不是只看哪条线最高。
6. 修改 `epsilon`、`c`、先验和 `step_size`，把观察写入 `notes/experiment.md`。

## 9. Definition of Done

- [x] M1：sample-average 与 constant-step-size 更新均有标准答案和数值测试。
- [x] M2：ε-greedy 含探索、利用和随机打破并列。
- [x] M3：UCB 与 Beta-Bernoulli Thompson Sampling 均已实现。
- [x] M4：主实验在同一 Bernoulli 环境比较三种策略。
- [x] M5：提供非平稳 Gaussian 实验理解固定步长。
- [x] M6：固定 seed 可复现；环境随机性和策略随机性相互独立。
- [x] M7：自动测试、CLI、自检与 Notebook 均可独立运行。

## 10. 常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 利用时 `np.argmax` 永远取第一个并列臂 | 下标偏置 | 在所有最大值动作中随机选 |
| 先采样奖励再计算非平稳 regret | 使用了漂移后的真实均值 | 在 `pull` 前记录 gap |
| 用 sampled reward 算 regret | 曲线可能下降或为负 | 使用真实期望的差，只用于评测 |
| UCB 未处理 `N(a)=0` | 除零 | 每臂先试一次 |
| Thompson 接受 0.3 之类连续奖励 | Beta-Bernoulli 后验失效 | 非 0/1 奖励立即报错 |
| 多次实验复用同一个 agent | 不同 run 不是独立重复 | factory 每次返回新对象 |

进一步推导和标准实验解读见 [notes/standard-answer.md](notes/standard-answer.md)。
