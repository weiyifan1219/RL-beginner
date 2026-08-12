# 任务二：MDP、Bellman 方程与动态规划

> Task 01 只有动作；本任务加入状态和时间。你将第一次完整看到“当前决策如何通过后继状态影响未来回报”。

## 一句话目标

用一个严格定义的表格型 MDP 和 4×4 GridWorld，从零实现 Bellman expectation backup、策略评估、策略迭代和值迭代，并把公式里的每一项对应到代码。

## 学完后你应该能回答

1. MDP 的 Markov 性质到底假设了什么？
2. `V^π`、`Q^π`、`V*` 和 `Q*` 分别回答什么问题？
3. Bellman 方程为什么是递归定义，却能通过迭代求解？
4. policy iteration 与 value iteration 的差别是什么？
5. 为什么真正终止的 transition 后面绝不能再 bootstrap？

## 1. MDP 定义

有限 MDP 用 `(S, A, P, R, γ)` 描述：

| 符号 | 含义 |
|---|---|
| `S` | 有限状态集合 |
| `A` | 有限动作集合 |
| `P(s',r\|s,a)` | 在 `(s,a)` 下到达 `s'` 并获得 `r` 的概率 |
| `R_t` | 一步奖励 |
| `γ ∈ [0,1]` | 对未来奖励的折扣 |

Markov 性质不是“未来与过去无关”，而是：给定当前状态和动作后，过去历史不再为下一步提供额外信息。

本仓库的转移表约定为：

```python
P[state][action] = [
    (probability, next_state, reward, terminated),
    ...,
]
```

每个 `(state, action)` 的分支概率之和必须为 1。`terminated=True` 表示 episode 在这次转移后真正结束，因此目标中没有 `γV(s')`。

## 2. Value 与 Bellman expectation

状态价值是从状态 `s` 开始并遵循策略 `π` 的期望折扣回报：

```text
V^π(s) = E_π[G_t | S_t=s]
```

将第一步拆出得到 Bellman expectation equation：

```text
V^π(s)
= Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)
  [r + γ(1-terminal)V^π(s')]
```

代码映射：

| 公式部分 | 标准答案 |
|---|---|
| 对一个 `(s,a)` 求期望 | `TabularMDP.expected_action_return` |
| 对策略动作概率求和 | `bellman_expectation_backup` |
| 反复 backup 到变化 `< theta` | `iterative_policy_evaluation` |

## 3. Bellman optimality

最优价值不再对策略求期望，而是选择最大的动作价值：

```text
V*(s) = max_a Σ_{s',r} p(s',r|s,a)
        [r + γ(1-terminal)V*(s')]
```

- **Policy iteration**：完整评估当前策略，再对 `Q(s,a)` 贪心改进，直到策略不再改变。
- **Value iteration**：每次直接做 Bellman optimality backup，可理解为把策略评估截短到一次。

并列最优动作不会武断取一个下标；标准答案在所有最大动作上均匀分配概率，所以策略仍是合法分布，也保留了所有最短路径。

当 `γ<1` 时 Bellman operator 是 contraction，固定点唯一。`γ=1` 需要额外条件：固定策略评估会验证该策略几乎必然终止；值迭代保守限定为每个状态存在终止路径、奖励非正且从零初始化的 shortest-path 问题。一般 continuing undiscounted MDP 不在本任务范围内。

## 4. GridWorld 约定

默认网格为 4×4，状态按 row-major 编号：

```text
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

| 项目 | 约定 |
|---|---|
| 动作 | `0=up, 1=right, 2=down, 3=left` |
| 终止状态 | 左上角 0、右下角 15 |
| 非终止奖励 | 每步 -1，包括撞墙 |
| 边界 | 撞墙留在原状态 |
| 终止状态 | 吸收状态，自环奖励 0，`terminated=True` |

均匀随机策略、`γ=1` 下的经典价值为：

```text
[[  0, -14, -20, -22],
 [-14, -18, -20, -20],
 [-20, -20, -18, -14],
 [-22, -20, -14,   0]]
```

最优价值是到最近终点的负最短步数：

```text
[[ 0, -1, -2, -3],
 [-1, -2, -3, -2],
 [-2, -3, -2, -1],
 [-3, -2, -1,  0]]
```

这两张表作为 golden tests，不只是看“算法是否收敛”，还验证环境语义是否一致。

## 5. 标准答案 API 与输入输出

### 构造 MDP

```python
from src.mdp import TabularMDP

mdp = TabularMDP([
    [
        [(1.0, 0, 1.0, False)],
        [(1.0, 1, 2.0, True)],
    ],
    [
        [(1.0, 1, 0.0, True)],
        [(1.0, 1, 0.0, True)],
    ],
])

one_step: float = mdp.expected_action_return(
    state=0, action=1, values=[10.0, 999.0], gamma=0.9
)
# 结果是 2.0；terminated=True，所以 999 不会被 bootstrap。
```

### 策略评估

```python
result = iterative_policy_evaluation(
    mdp,
    policy,                 # float64, shape (n_states, n_actions)，每行和为 1
    gamma=0.9,
    theta=1e-10,
)

result.values       # float64, shape (n_states,)
result.iterations   # 执行的 sweep 次数
result.converged    # 是否在 max_iterations 前达到 theta
result.delta        # 最后一轮 max_s |V_new(s)-V_old(s)|
```

### 最优控制

```python
pi = policy_iteration(env, gamma=1.0, theta=1e-10)
vi = value_iteration(env, gamma=1.0, theta=1e-10)

vi.policy  # float64, shape (S,A)，并列最优动作均分概率
vi.values  # float64, shape (S,)
```

## 6. 文件地图

| 文件 | 内容 |
|---|---|
| `src/mdp.py` | Transition 与严格校验的 TabularMDP |
| `src/gridworld.py` | 确定性 4 邻域 GridWorld |
| `src/dynamic_programming.py` | 策略评估、改进、PI、VI |
| `src/visualization.py` | 价值热图和随机策略箭头 |
| `run_experiment.py` | YAML 驱动求解与结果保存 |
| `tests/` | 解析 MDP、终止 mask、golden values、CLI 契约 |
| `eval/run.py` | 一键自检并生成 `eval/result.json` |
| `notebooks/task-02-mdp-dp.ipynb` | 可执行中文讲义 |
| `notes/standard-answer.md` | 推导和实现设计说明 |

## 7. 运行方式

```bash
cd /workspace/YiFan/llm_agent/repos/RL-beginner
conda activate llm-agent

python task-02-mdp-dp/eval/run.py

python task-02-mdp-dp/run_experiment.py --quick \
  --output-dir task-02-mdp-dp/outputs/quick

python task-02-mdp-dp/run_experiment.py \
  --algorithm policy_iteration \
  --output-dir task-02-mdp-dp/outputs/policy-iteration

cd task-02-mdp-dp
python -m jupyter lab
```

Notebook 请选择 `python3` 内核。CLI 输出：

```text
outputs/<run>/
├── resolved_config.yaml
├── summary.json
├── values.npy
├── policy.npy
└── value_and_policy.png
```

## 8. 建议学习顺序

1. 在纸上对一个两状态 MDP 手算一次 backup。
2. 先实现 `expected_action_return`，用 `terminated=True` 的极大后继价值检验 mask。
3. 完成 policy evaluation，并复现 4×4 随机策略 golden values。
4. 再写 greedy improvement 和 policy iteration。
5. 最后把两步压缩成 value iteration，比较两者 sweeps 和输出策略。
6. 顺序执行 Notebook，修改 `γ`、`theta` 和网格大小，记录解释。

## 9. Definition of Done

- [x] M1：转移表概率、索引、奖励和终止标记有严格检查。
- [x] M2：Bellman expectation backup 与解析小 MDP 数值一致。
- [x] M3：策略评估复现 Sutton 4×4 GridWorld golden values。
- [x] M4：策略迭代和值迭代得到相同最优价值与随机最优策略。
- [x] M5：终止转移不 bootstrap，非收敛会显式报告。
- [x] M6：测试、CLI、自检和 Notebook 可独立运行。

## 10. 常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| `terminated` 后仍加 `γV(s')` | 终点价值泄漏到目标 | 乘 `(1-terminated)` |
| 转移概率未检查为 1 | backup 不是合法期望 | 构造 MDP 时失败 |
| 原地更新却按同步公式解释 | sweep 轨迹不同 | 明确实现；本答案使用同步副本 |
| `argmax` 随便丢掉并列最优动作 | 策略依赖下标 | 所有最大动作均分概率 |
| 只写 `while True` 等收敛 | 错误 MDP 可能死循环 | 同时设置 theta 和 max_iterations |
| 把 terminal state 与“撞墙不动”混淆 | 边界提前结束 | 只有进入角落才 terminated |

完整推导见 [notes/standard-answer.md](notes/standard-answer.md)。
