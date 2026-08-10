# 术语表

| 术语 | 含义 |
|---|---|
| Agent | 在状态下选动作的决策主体 |
| Environment | 接受动作、返回新状态和奖励的系统 |
| MDP | 用 `(S, A, P, R, γ)` 描述的序贯决策过程 |
| On-policy | 用当前策略采样并学习，如 SARSA、PPO |
| Off-policy | 可复用其他策略采样的数据，如 DQN、SAC |
| Bootstrap | 用下一状态的当前估计构造学习目标 |
| Replay buffer | 保存过去 transition 并随机采样的经验池 |
| Rollout | 用策略与环境交互得到的一段轨迹 |
| Advantage | 动作相对状态平均水平的好坏，通常为 `Q-V` |
| Offline RL | 只使用固定数据集、不能或很少在线探索的 RL |
| RLHF | 用人类偏好/奖励训练模型并以 RL 优化的范式 |
| DPO | 直接以偏好对优化策略的方法 |

