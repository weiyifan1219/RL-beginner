# RL-Beginner Standard Solutions Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** 将 10 个强化学习任务逐一建设为包含标准答案、明确输入输出、自动测试、实验脚本、中文教学文档和可执行 Jupyter Notebook 的完整课程。

**Architecture:** 每个任务保持同一学习闭环：`README` 讲原理和接口，`src` 给出透明的参考实现，`tests` 固定行为契约，`eval/run.py` 给学习者一键自检，`run_experiment.py` 产出可复现实验，`notebooks` 串联公式、代码和图表。经典 RL 任务不依赖高层 RL 框架；任务 9–10 才引入已有本地 Qwen、偏好训练和 rollout 基础设施，并保证无模型时仍可运行核心单元测试。

**Tech Stack:** Python 3.10+、NumPy、PyTorch、Gymnasium、Matplotlib、TensorBoard、PyYAML、pytest、Jupyter/nbclient。

---

## 统一完成标准

每个任务进入下一任务前必须满足：

1. 所有公开类和函数都有类型标注、docstring 和明确的输入输出形状。
2. 新行为先写测试并确认因缺少实现而失败，再写最小实现使其通过。
3. `python eval/run.py` 输出结构化 `eval/result.json`，失败时退出码非零。
4. `python run_experiment.py --quick` 能在合理时间内产出 JSON 指标和至少一张图。
5. Notebook 从干净 kernel 顺序执行成功，不依赖手工预设变量或外部下载。
6. README 包含公式、实现映射、运行命令、预期现象、常见坑和 DoD。
7. 在 3090 的 `/root/miniconda3/envs/llm-agent` 中完成验证，然后从远端工作副本提交。

## 统一目录

```text
task-XX-name/
├── README.md
├── requirements.txt
├── configs/default.yaml
├── src/
│   ├── __init__.py
│   └── ...
├── tests/
│   └── test_*.py
├── eval/
│   ├── run.py
│   └── tutor_prompt.md
├── notebooks/
│   └── task-XX-*.ipynb
├── run_experiment.py
├── figures/.gitkeep
├── data/.gitkeep
└── notes/
    ├── experiment.md
    └── standard-answer.md
```

## Phase 1：经典强化学习基础

### Task 01：Bandit 与探索

**Files:**

- Create: `task-01-bandit/src/bandits.py`
- Create: `task-01-bandit/src/agents.py`
- Create: `task-01-bandit/src/experiment.py`
- Create: `task-01-bandit/tests/test_bandits.py`
- Create: `task-01-bandit/tests/test_agents.py`
- Create: `task-01-bandit/tests/test_experiment.py`
- Create: `task-01-bandit/run_experiment.py`
- Create: `task-01-bandit/notebooks/task-01-bandit.ipynb`
- Modify: `task-01-bandit/eval/run.py`
- Modify: `task-01-bandit/README.md`

**Contract:** 实现 Gaussian/Bernoulli bandit、sample-average/constant-step estimator、ε-greedy、UCB、Bernoulli Thompson Sampling；统一实验返回 `(n_steps,)` 的平均奖励、最优动作率、即时和累计期望 regret，并保证同 seed 可复现。

**Verification:** `pytest -q task-01-bandit/tests`、`python task-01-bandit/eval/run.py`、quick CLI、Notebook 干净执行。

### Task 02：MDP 与动态规划

**Files:** `src/mdp.py`、`src/gridworld.py`、`src/dynamic_programming.py`、三组 `tests/test_*.py`、实验 CLI、Notebook、README、standard answer。

**Contract:** 明确 `P[s][a] -> [(prob, next_state, reward, terminated)]`；实现 Bellman expectation backup、迭代 policy evaluation、policy improvement、policy iteration、value iteration。解析小 MDP 和 GridWorld 的最优值/策略作为 golden tests。

**Verification:** 概率和为 1、终止状态不 bootstrap、两种最优算法策略一致、Notebook 输出价值表和策略箭头。

### Task 03：表格型 MC/TD/SARSA/Q-Learning

**Files:** `src/prediction.py`、`src/control.py`、`src/policies.py`、`src/train.py`、tests、CLI、Notebook、README。

**Contract:** first-visit MC 与 TD(0) 预测；ε-greedy SARSA 与 Q-Learning 控制；正确区分 `terminated`/`truncated`；训练返回 episode return、length、Q table。

**Verification:** tiny deterministic MDP 的精确更新值；FrozenLake 学习成功率；CliffWalking 中 SARSA/Q-Learning 的路径风险差异。

## Phase 2：深度强化学习

### Task 04：DQN

**Files:** `src/replay_buffer.py`、`src/network.py`、`src/dqn.py`、`src/train.py`、tests、CartPole CLI/Notebook、README。

**Contract:** replay buffer、online/target Q network、DQN TD target、ε schedule、硬同步；进阶 Double DQN。网络输入 `(B, obs_dim)`，输出 `(B, n_actions)`。

**Verification:** buffer 边界、terminal mask、target 无梯度、同步精确一致；quick smoke 与固定预算 CartPole 学习检查。

### Task 05：Policy Gradient

**Files:** `src/policy.py`、`src/returns.py`、`src/reinforce.py`、tests、CLI/Notebook、README。

**Contract:** categorical policy、折扣 return、reward-to-go、REINFORCE loss、可选 value baseline/entropy；批次维度和时间维度清晰。

**Verification:** 手算轨迹 return 和 loss；梯度方向检查；多 seed 比较 baseline 对方差的影响。

### Task 06：Actor-Critic 与 GAE

**Files:** `src/models.py`、`src/rollout.py`、`src/gae.py`、`src/a2c.py`、tests、CLI/Notebook、README。

**Contract:** actor/critic 联合或分离网络、rollout buffer、TD residual、倒序 GAE、policy/value/entropy loss；对 time-limit truncation 允许 bootstrap。

**Verification:** 人工序列的 GAE 精确值；λ=0 等价 TD residual，λ=1 对应 Monte Carlo 型优势；CartPole smoke。

### Task 07：PPO

**Files:** `src/models.py`、`src/rollout.py`、`src/ppo.py`、tests、CLI/Notebook、README。

**Contract:** 缓存 old log-prob/value、clipped surrogate、clipped/unclipped value loss、entropy、mini-batch 多 epoch、approx KL 与 clip fraction。

**Verification:** ratio=1、正负 advantage 的 clip 边界精确测试；old tensors 不带梯度；quick 训练和 checkpoint resume。

### Task 08：SAC

**Files:** `src/networks.py`、`src/replay_buffer.py`、`src/sac.py`、tests、Pendulum CLI/Notebook、README。

**Contract:** tanh-squashed Gaussian actor、log-prob Jacobian 修正、twin critic、soft target、actor/critic/automatic-alpha 更新；动作映射到环境上下界。

**Verification:** shape/range、有限 log-prob、Polyak 更新精确值、target 无梯度、Pendulum quick smoke。

## Phase 3：离线对齐、Agentic RL 与 Infra

### Task 09：Offline RL、RLHF 与 DPO

**Files:** `src/dataset.py`、`src/behavior_cloning.py`、`src/cql.py` 或 `src/iql.py`、`src/dpo.py`、`src/model_locator.py`、tests、toy CLI/Notebook、README。

**Contract:** 固定 transition dataset；BC baseline；至少一个可读的保守离线 RL 算法；纯张量 DPO loss 与 chosen/rejected log-prob；模型定位优先级 `RL_BEGINNER_MODEL_PATH` → `LLM_BEGINNER_MODEL_PATH` → `/workspace/YiFan/llm_agent/models` → 相邻 `llm-beginner`，禁止隐式下载。

**Verification:** OOD action 诊断、保守惩罚方向、DPO 对手算值、无模型测试全通过；可选 Qwen LoRA smoke 明确标记 skip 原因。

### Task 10：MiniRL、Agentic RL 与训练基础设施

**Files:** `minirl/config.py`、`envs.py`、`policy.py`、`buffers.py`、`rollout.py`、`learner.py`、`logger.py`、`checkpoint.py`、`agentic/verifier.py`、tests、CLI/Notebook、README。

**Contract:** 统一 Env/Policy/Buffer/Learner 协议；vector rollout；JSONL/TensorBoard logging；原子 checkpoint/resume；以 CartPole 验证 infra；可选用本地 Qwen 在可验证任务上收集 prompt/response/reward trace。

**Verification:** 多环境 shape、seed 可复现、checkpoint 恢复后参数/optimizer/step 一致、日志 schema、verifier 确定性；Notebook 展示从单环境到 agentic rollout 的数据流。

## 提交顺序

每个任务至少一个独立提交：`feat(task-XX): add complete ... solution`。只有当前任务的测试、评测、实验和 Notebook 全部通过后才开始下一个任务；禁止把多个未验证任务堆在一个提交中。
