# 任务五：Policy Gradient

> 主路线见仓库根 [README](../README.md)。本任务是一个可逐步填充的学习单元：先推导，再在 `src/` 中手写，再用实验解释现象，最后才对照成熟框架。

## 一句话目标

直接参数化随机策略，在 REINFORCE 中推导 log-derivative trick，并通过 baseline 和 return normalization 观察方差控制。

## 核心概念

likelihood ratio；trajectory return；baseline；熵正则

## 本任务交付

实现离散动作 categorical policy、采样轨迹、REINFORCE loss；在 CartPole 对比无 baseline 与 value baseline。

## Definition of Done

- [ ] M1：正确计算折扣回报
- [ ] M2：实现 REINFORCE
- [ ] M3：实现 baseline
- [ ] M4：记录至少 5 个 seed 的均值/方差
- [ ] M5：解释高方差来源。

## 建议步骤

1. 在 `notes/` 手写或整理关键公式，标明每一个期望来自何处。
2. 先创建最小的、可复现的环境与随机种子，完成 `src/` 的朴素实现。
3. 增加训练循环和评测循环，输出曲线到 `figures/`、指标到 `notes/`。
4. 做至少一个只改动一个变量的对照实验，并解释结果。
5. 最后阅读对应框架实现或 Stable-Baselines3 / TRL 文档，对照而不替代手写版本。

## 前置知识

任务 4；math-basics 中 log-derivative trick。

## 实现边界与常见提醒

先验证同一轨迹的 loss 符号，再看学习曲线。

## 当前目录状态

本仓库当前只建立教学骨架，尚未提供标准答案或完整 `eval/run.py`。后续按学习进度补充本任务的精确接口、数值单测和训练脚本；现在可运行：

```bash
python eval/run.py
```

它会确认任务骨架已就绪。

## 推荐记录格式

在 `notes/experiment.md` 中记录：日期、Git commit、环境版本、种子、超参、训练步数、最终指标、曲线位置、一个失败现象和你的解释。

