# 任务十：MiniRL、Agentic RL 与训练基础设施

> 主路线见仓库根 [README](../README.md)。本任务是一个可逐步填充的学习单元：先推导，再在 `src/` 中手写，再用实验解释现象，最后才对照成熟框架。

## 一句话目标

把前面手写的算法整理为一个小而透明的训练系统，并探索 LLM/agent 的 rollout、verifier reward、并行采样和可观测性。

## 核心概念

统一接口；vector env；rollout worker；checkpoint；可复现；verifier reward

## 本任务交付

构建 MiniRL 最小接口（Env/Policy/Buffer/Learner/Logger），将 PPO 或 SAC 接入；可选把本地 Qwen 作为 policy，在固定可验证任务上收集 agentic rollout。

## Definition of Done

- [ ] M1：统一 config 与 seed
- [ ] M2：vectorized rollout
- [ ] M3：TensorBoard/JSONL 日志
- [ ] M4：原子 checkpoint/resume
- [ ] M5：最小 agentic verifier 任务与 reward trace。

## 建议步骤

1. 在 `notes/` 手写或整理关键公式，标明每一个期望来自何处。
2. 先创建最小的、可复现的环境与随机种子，完成 `src/` 的朴素实现。
3. 增加训练循环和评测循环，输出曲线到 `figures/`、指标到 `notes/`。
4. 做至少一个只改动一个变量的对照实验，并解释结果。
5. 最后阅读对应框架实现或 Stable-Baselines3 / TRL 文档，对照而不替代手写版本。

## 前置知识

任务 1–9；任务 9 的本地模型/服务配置可直接复用。

## 实现边界与常见提醒

先以 CartPole 验证 infra，再接 LLM。明确区分训练数据、模型权重、日志与代码，不提交大文件。

## 当前目录状态

本仓库当前只建立教学骨架，尚未提供标准答案或完整 `eval/run.py`。后续按学习进度补充本任务的精确接口、数值单测和训练脚本；现在可运行：

```bash
python eval/run.py
```

它会确认任务骨架已就绪。

## 推荐记录格式

在 `notes/experiment.md` 中记录：日期、Git commit、环境版本、种子、超参、训练步数、最终指标、曲线位置、一个失败现象和你的解释。

