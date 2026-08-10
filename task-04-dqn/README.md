# 任务四：DQN 与稳定的价值学习

> 主路线见仓库根 [README](../README.md)。本任务是一个可逐步填充的学习单元：先推导，再在 `src/` 中手写，再用实验解释现象，最后才对照成熟框架。

## 一句话目标

从表格 Q-Learning 推进到函数逼近：在 CartPole 上手写 DQN，并通过 replay buffer 与 target network 观察稳定性差异。

## 核心概念

函数逼近；experience replay；target network；ε schedule；过估计

## 本任务交付

实现 MLP Q-network、replay buffer、训练/评估分离和 TensorBoard 曲线；进阶实现 Double DQN。

## Definition of Done

- [ ] M1：replay buffer 随机采样
- [ ] M2：DQN TD target
- [ ] M3：周期性 target 同步
- [ ] M4：CartPole 达到预先约定阈值
- [ ] M5：移除稳定化组件的失败对照。

## 建议步骤

1. 在 `notes/` 手写或整理关键公式，标明每一个期望来自何处。
2. 先创建最小的、可复现的环境与随机种子，完成 `src/` 的朴素实现。
3. 增加训练循环和评测循环，输出曲线到 `figures/`、指标到 `notes/`。
4. 做至少一个只改动一个变量的对照实验，并解释结果。
5. 最后阅读对应框架实现或 Stable-Baselines3 / TRL 文档，对照而不替代手写版本。

## 前置知识

任务 3、PyTorch autograd。

## 实现边界与常见提醒

不要直接使用 Stable-Baselines3；先保障 target 张量不参与梯度。

## 当前目录状态

本仓库当前只建立教学骨架，尚未提供标准答案或完整 `eval/run.py`。后续按学习进度补充本任务的精确接口、数值单测和训练脚本；现在可运行：

```bash
python eval/run.py
```

它会确认任务骨架已就绪。

## 推荐记录格式

在 `notes/experiment.md` 中记录：日期、Git commit、环境版本、种子、超参、训练步数、最终指标、曲线位置、一个失败现象和你的解释。

