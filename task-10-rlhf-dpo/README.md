# Task 10：LLM Alignment——RLHF & DPO

## 一句话目标

在一个任务内理解大模型偏好对齐的两条主线：经典 SFT → Reward Model → PPO RLHF，以及从 KL-Regularized RL 推导出的 DPO。

## 学习范围

### 经典 RLHF

- SFT、demonstration data 与 instruction-following policy。
- chosen / rejected preference data 与 Bradley–Terry preference model。
- Reward Model 的标量奖励、排序目标和验证方式。
- Reference Model、KL penalty、token-level reward shaping。
- Advantage / GAE、response mask 和 PPO clipped objective。
- InstructGPT 风格三阶段 pipeline 的数据流。

### DPO

- KL-Regularized RL objective 与最优 policy。
- Reference policy、log-ratio、chosen/rejected response 和 DPO loss。
- β、preference noise、response length 对训练的影响。
- DPO 为什么不需要显式 Reward Model + PPO，以及相应代价。
- IPO、KTO 等方法只做思想层面对比。

## 完成标准

- [ ] 画出 RLHF 与 DPO 的数据流、模型关系和差异。
- [ ] 在 toy preference 数据上验证 Bradley–Terry loss 和 reward ranking。
- [ ] 在纯张量数据上验证 DPO loss 的符号、梯度方向和 reference 作用。
- [ ] 将 PPO 的 advantage、ratio、clip 和 KL 对应到 LLM token。
- [ ] 完成一个小规模、可复现的 toy alignment 实验。

## 目录用途

- src/：SFT、preference、Reward Model、PPO 和 DPO 的最小实现。
- notebooks/01-learning.ipynb：pipeline、推导和 toy tensor 验证。
- notebooks/02-experiments.ipynb：KL、β、长度和 preference noise 实验。
- eval/：loss、mask、KL、优势估计和对齐接口检查。
- figures/：RLHF/DPO 对照、reward、KL、loss 和长度曲线。
- notes/：公式推导、阅读笔记、数据 schema 和实验记录。

## 边界

本任务聚焦偏好对齐基础，不扩展到在线 rollout、GRPO 或 RLVR；这些内容放在 Task 11。优先使用 toy model、已有本地模型或兼容服务。

