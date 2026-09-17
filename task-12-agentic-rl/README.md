# Task 12：Agentic RL & LLM RL Engineering

## 一句话目标

把前面学到的 RL、LLM rollout、verifier reward 组织成一个可观测、可恢复、可扩展的 Agentic RL training pipeline，完成整个学习路线的工程收尾。

## 学习范围

- Multi-turn RL、tool-use RL 和完整 agent trajectory。
- Environment interaction、sparse/delayed reward 与 credit assignment。
- observation、action、tool result、termination 和 reward trace。
- rollout worker、并行采样、batching、日志、checkpoint 和 resume。
- TRL、verl、vLLM rollout 等工程框架的职责边界。
- 从单机教学代码到真实 LLM RL training pipeline 的迁移清单。

## 完成标准

- [ ] 定义可序列化的 multi-turn agent trajectory schema。
- [ ] 在 toy tool environment 中记录 sparse/delayed reward 和 credit assignment。
- [ ] 设计 rollout、learner、verifier、logger、checkpoint 的模块边界。
- [ ] 完成一次最小 agentic RL 闭环，并验证中断后可以 resume。
- [ ] 对照 TRL / verl / vLLM，写出单机到分布式实现的差异。

## 目录用途

- src/：trajectory、environment、rollout worker、learner 和 checkpoint 接口。
- notebooks/01-learning.ipynb：agent trajectory、reward trace 和模块关系。
- notebooks/02-experiments.ipynb：工具调用、延迟奖励、并行 rollout 和 resume 实验。
- eval/：trajectory schema、tool interaction、reward trace、checkpoint/resume 检查。
- figures/：agent workflow、时序 reward、吞吐和训练曲线。
- notes/：框架对照、工程决策、故障记录和最终复盘。

## 边界

这是最终收尾任务，不再继续拆分新的算法 Task。先完成单机、可解释、可恢复的最小闭环，再考虑多 GPU、分布式 rollout 或更大模型。

