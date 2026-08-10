# 任务八：SAC 与连续控制

> 主路线见仓库根 [README](../README.md)。本任务是一个可逐步填充的学习单元：先推导，再在 `src/` 中手写，再用实验解释现象，最后才对照成熟框架。

## 一句话目标

在 Pendulum 上手写 Soft Actor-Critic，理解最大熵目标、reparameterization、twin Q 与自动温度调节。

## 核心概念

连续动作；高斯策略；reparameterization；twin critics；entropy temperature

## 本任务交付

实现 squashed Gaussian actor、两个 critic、replay buffer 和 soft target update；对比固定与自动 α。

## Definition of Done

- [ ] M1：动作缩放到环境边界
- [ ] M2：twin-Q target
- [ ] M3：soft target update
- [ ] M4：actor/critic/α 三类更新
- [ ] M5：Pendulum 曲线与温度曲线。

## 建议步骤

1. 在 `notes/` 手写或整理关键公式，标明每一个期望来自何处。
2. 先创建最小的、可复现的环境与随机种子，完成 `src/` 的朴素实现。
3. 增加训练循环和评测循环，输出曲线到 `figures/`、指标到 `notes/`。
4. 做至少一个只改动一个变量的对照实验，并解释结果。
5. 最后阅读对应框架实现或 Stable-Baselines3 / TRL 文档，对照而不替代手写版本。

## 前置知识

任务 4、6；连续分布与重参数化。

## 实现边界与常见提醒

对 log-prob 的 tanh 修正做单元数值检查；不要混淆 reward 和 entropy 项的符号。

## 当前目录状态

本仓库当前只建立教学骨架，尚未提供标准答案或完整 `eval/run.py`。后续按学习进度补充本任务的精确接口、数值单测和训练脚本；现在可运行：

```bash
python eval/run.py
```

它会确认任务骨架已就绪。

## 推荐记录格式

在 `notes/experiment.md` 中记录：日期、Git commit、环境版本、种子、超参、训练步数、最终指标、曲线位置、一个失败现象和你的解释。

