# Task 09：RL Review & Bridge to LLM RL

## 一句话目标

统一复盘 Task 1–8 的经典强化学习知识，并把传统 RL 的对象、目标和训练循环映射到 LLM generation，为后续大模型 RL 学习做准备。

## 学习范围

- TD(λ)、λ-return 与 GAE。
- Behavior Cloning / Imitation Learning。
- Offline RL 与 Online RL 的数据分布、OOD action 和交互成本。
- CQL / IQL 的核心思想，不做重型实验。
- Entropy Regularization、KL Regularization 与 KL-Regularized RL。
- LLM generation 的 state、action、policy、trajectory、reward 建模。
- BC ↔ SFT、policy ↔ LLM、trajectory ↔ completion 的对应关系。

## 完成标准

- [x] 完成 Task 1–8 核心知识复盘。
- [x] 能从 TD(0) 写出 TD(λ) 和 GAE 的递推关系。
- [x] 能区分 entropy bonus 与 KL penalty。
- [x] 能把一次 LLM completion 写成标准 RL trajectory。
- [x] 完成不依赖大模型的 toy bridge 验证。

## 目录用途

- src/：TD(λ)、GAE、BC 和 toy regularized-RL 实现。
- notebooks/01-learning.ipynb：公式推导、对象映射和最小验证。
- notebooks/02-experiments.ipynb：λ、entropy/KL 权重和数据分布实验。
- eval/：递推公式、边界条件和 trajectory schema 检查。
- figures/：GAE、distribution shift 和 regularization 曲线。
- notes/：复盘表、LLM-RL 对照表和实验记录。

## 边界

本任务不训练大模型。完整的 Reward Model、PPO-RLHF 和 DPO 合并到 Task 10；在线 rollout 和 RLVR 放到 Task 11。

