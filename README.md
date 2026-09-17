# RL-Beginner：强化学习到 LLM RL 的 12 个任务

本仓库面向已经掌握 Python、PyTorch 和基础深度学习的学习者，采用“公式与直觉 → 手写实现 → 小型实验 → 工程对照”的学习路径。

路线固定为 12 个 Task。Task 1–8 完成经典强化学习基础，Task 9 完成传统 RL 到 LLM RL 的桥接，Task 10–12 集中学习大模型对齐、在线推理强化学习、RLVR 和 Agentic RL，避免继续无边界扩展算法数量。

## 路线图

| Task | 主题 | 核心内容 | 当前状态 |
|---|---|---|---|
| 01 | Multi-Armed Bandit | ε-greedy、UCB、Thompson Sampling、regret | ✅ 已完成 |
| 02 | MDP / Dynamic Programming | Bellman 方程、Policy Iteration、Value Iteration | ✅ 已完成 |
| 03 | MC / TD / SARSA / Q-Learning | model-free prediction 与 control | ✅ 已完成 |
| 04 | DQN | replay buffer、target network、稳定训练 | ✅ 已完成 |
| 05 | REINFORCE / Policy Gradient | policy gradient、baseline、return | ✅ 已完成 |
| 06 | Actor-Critic / n-step | critic、n-step return、优势估计 | ✅ 已完成 |
| 07 | TRPO / PPO | trust region、clipped objective、GAE | ✅ 已完成 |
| 08 | DDPG / TD3 / SAC | 连续动作、twin Q、最大熵 RL | ✅ 已完成 |
| 09 | RL Review & Bridge to LLM RL | TD(λ)、GAE、BC/IL、Offline RL、KL-Regularized RL、LLM trajectory 建模 | ✅ 已完成 |
| 10 | LLM Alignment：RLHF & DPO | SFT、Preference Data、Reward Model、PPO、KL、DPO | 🧱 下一步 |
| 11 | Online LLM RL & RLVR | RLOO、GRPO、online rollout、verifiable reward、reasoning RL | 🧱 待学习 |
| 12 | Agentic RL & Engineering | multi-turn、tool use、trajectory、delayed reward、rollout worker、TRL/verl/vLLM | 🧱 收尾任务 |

## 目录结构

RL-beginner/
├── task-01-bandit/
├── task-02-mdp-dp/
├── task-03-tabular-rl/
├── task-04-dqn/
├── task-05-policy-gradient/
├── task-06-actor-critic/
├── task-07-trpo_ppo/
├── task-08-sac/
├── task-09-llm-bridge/
├── task-10-rlhf-dpo/
├── task-11-online-rlvr/
├── task-12-agentic-rl/
├── eval/
├── requirements.txt
└── README.md

每个任务统一包含：

task-XX-topic/
├── src/                  # 手写实现
├── notebooks/            # 01-learning：概念；02-experiments：实验
├── notes/                # 公式、阅读笔记、实验记录和复盘
├── figures/              # 实验曲线、表格或可视化结果
├── eval/                 # 任务专用自检
├── data/                 # 小型数据或 schema，不放模型和大文件
└── README.md             # 任务目标、学习顺序与完成标准

## Task 9–12 的学习边界

- Task 9 只负责复盘和建立 LLM generation 的 RL 表达，不重复实现经典 RL。
- Task 10 将经典 RLHF 与 DPO 放在同一个对齐任务中，重点理解两条偏好优化路线的关系。
- Task 11 将现代在线 LLM RL 与 RLVR 放在一起，连接 group-relative advantage、verifier reward 和 reasoning。
- Task 12 是工程收尾，不再拆分新的算法 Task；重点是 agent trajectory、环境交互、可观测性和可恢复训练。
- 优先完成 toy / tensor-level 验证，再接入已有本地模型或兼容服务；不默认下载大模型。

## 通用学习循环

1. 阅读任务 README，在 notes/ 中先写公式、假设和待回答问题。
2. 在 src/ 中完成最小实现，让 Notebook 调用 src/，不要复制另一套算法。
3. 在 eval/ 中补充数值、接口和端到端 smoke test。
4. 将实验产物写入 figures/，把数值和解释记录在 notes/experiment.md。
5. 最后对照 TRL、verl、vLLM 等框架，记录手写版本与工程实现的差异。

## 环境

默认在 3090 服务器的 llm-agent 环境中工作：

    cd /workspace/YiFan/llm_agent/repos/RL-beginner
    python _eval_harness.py --check-env

不要为了本仓库重新下载大模型。需要模型时，优先复用已有模型目录或 OpenAI 兼容推理服务。

## 许可证

[MIT License](LICENSE)。
