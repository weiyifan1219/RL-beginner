# Task 11：Online LLM RL & RLVR

## 一句话目标

把现代在线 LLM RL 与推理强化学习放在同一个闭环中，理解 RLOO、GRPO、critic-free learning 如何利用 rollout 分组和可验证奖励训练推理策略。

## 学习范围

### Online LLM RL

- REINFORCE 在 LLM 中重新出现的原因。
- Online rollout、采样分组和即时/延迟 reward。
- RLOO 的 leave-one-out baseline。
- GRPO 的 group-relative advantage 与 critic-free 更新。
- PPO、RLOO、GRPO 的数据流、方差和显存取舍。
- DAPO / GSPO 作为改进思想对比，不逐个独立实现。

### Reasoning RL / RLVR

- Outcome Reward、Verifiable Reward 和 Rule-based Reward。
- 数学、Code、Logic 任务的 verifier 设计。
- RLVR 的采样、验证、奖励、更新和评估闭环。
- DeepSeek-R1 类训练范式的核心思想与必要限定。
- Reward hacking、长度偏置、格式投机和 verifier 漏洞。
- 探索、有效轨迹比例与能力变化的关系。

## 完成标准

- [ ] 能推导 group-relative advantage 和 leave-one-out baseline。
- [ ] 能说明 online rollout、reference policy、reward/verifier 的接口关系。
- [ ] 在 toy completion 环境中比较 group size 或 baseline 的方差。
- [ ] 为数学、Code 或 Logic toy 任务定义可验证 reward。
- [ ] 完成一个小型 reasoning RL 实验，并检查一种 reward hacking 或长度偏置。

## 目录用途

- src/：rollout schema、group sampler、RLOO/GRPO、verifier 和 reward parser。
- notebooks/01-learning.ipynb：推导、单组 rollout 和 verifier 验证。
- notebooks/02-experiments.ipynb：group size、temperature、reward variance 和长度实验。
- eval/：group mask、advantage、verifier 正确性和 reward 一致性检查。
- figures/：advantage、reward、accuracy、length 和失败样例统计。
- notes/：RLOO/GRPO/RLVR 对照、任务协议和实验记录。

## 边界

本任务优先使用 toy online rollout 和小模型验证，不声称复现大型推理模型训练规模。Agent trajectory、tool use 和工程框架放到 Task 12。

