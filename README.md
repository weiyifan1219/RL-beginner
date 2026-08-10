# RL-Beginner：强化学习任务驱动入门

面向已掌握 Python、PyTorch 和基础深度学习的学习者。本仓库按**原理与公式 → 手写实现 → 实验观察 → 成熟框架对照**的路径，逐步建立从 MDP 到大模型对齐和 RL 工程的完整认知。

## 路线图

| 任务 | 主题 | 主要问题 | 阶段产出 |
|---|---|---|---|
| 1 | 多臂老虎机 | 探索与利用如何权衡？ | bandit 模拟器、ε-greedy/UCB/Thompson 对比 |
| 2 | MDP 与动态规划 | Bellman 方程如何求解最优策略？ | policy/value iteration、GridWorld |
| 3 | 表格型 RL | 没有模型时如何从经验学习？ | MC、TD(0)、SARSA、Q-Learning |
| 4 | DQN | 如何用神经网络近似 Q 函数？ | replay buffer、target network、CartPole |
| 5 | Policy Gradient | 如何直接优化随机策略？ | REINFORCE、baseline、回报归一化 |
| 6 | Actor-Critic 与 GAE | 如何降低策略梯度方差？ | A2C、GAE、优势估计实验 |
| 7 | PPO | 如何稳定地进行策略更新？ | clipped objective、rollout、MiniBatch 更新 |
| 8 | SAC | 熵正则和连续控制如何结合？ | twin Q、温度调节、Pendulum |
| 9 | Offline RL 与 DPO | 离线数据和偏好数据怎样替代在线奖励？ | CQL/IQL 概念实验、Qwen DPO 复用接口 |
| 10 | MiniRL、Agentic 与 Infra | 算法如何变成可训练、可观测的系统？ | vector env、rollout worker、日志/检查点、agentic RL 实验 |

> 前 1–8 个任务以经典控制环境和小型实验为主；任务 9–10 再连接到本地 Qwen、偏好优化、agentic rollout 与训练基础设施。默认不下载大模型，优先复用已有本地模型目录或 OpenAI 兼容推理服务。

## 教学约定

每个任务目录均包含：

| 目录/文件 | 用途 |
|---|---|
| `README.md` | 原理、DoD、步骤、实验与接口约定 |
| `src/` | 学习者手写算法的位置 |
| `data/` | 小数据、离线数据或环境配置；不提交大文件 |
| `eval/` | 后续任务补充的自检与 tutor prompt |
| `figures/` | 曲线、策略图、轨迹图等实验结果 |
| `notes/` | 实验观察、推导笔记和复盘 |

根目录 `_eval_harness.py` 提供统一的最小评测壳。现在它只做结构与环境检查；随着每个任务逐一实现，再把该任务的数值契约加入对应 `eval/run.py`。

## 环境

推荐复用现有 `llm-agent` Conda 环境，Python >= 3.10（服务器上的该环境为 Python 3.11）：

```bash
conda activate llm-agent
pip install -r requirements.txt
python _eval_harness.py --check-env
```

`torch` 请按 3090 的 CUDA 环境选择官方对应 wheel；不要为了本仓库重新下载 Qwen 或其他大模型。任务 9 会依次从 `RL_BEGINNER_MODEL_PATH`、`LLM_BEGINNER_MODEL_PATH` 以及既有 `llm-beginner` 模型目录寻找可复用模型；也可使用 `OPENAI_BASE_URL` 指向本地服务。

## 通用学习循环

```bash
cd task-1-bandit
# 1. 阅读 README 中的公式和预备问题
# 2. 在 src/ 手写实现
# 3. 运行任务自检（该任务实现后提供）
python eval/run.py
# 4. 保存 figures/ 与 notes/ 中的观察
```

1. 先自己推导 Bellman backup、return、advantage 和目标函数，再看代码。
2. 先完成最小版本，再做一个可解释的消融；不要一开始套 Stable-Baselines3。
3. 每次实验至少记录随机种子、环境版本、超参、曲线和失败现象。
4. 完成手写版本后，才用 Gymnasium、Stable-Baselines3、TRL 等框架对照 API 与工程取舍。

## 文档

- [强化学习速查表](docs/rl-cheatsheet.md)
- [数学预备知识](docs/math-basics.md)
- [术语表](docs/glossary.md)

## 许可证

[MIT License](LICENSE)。

