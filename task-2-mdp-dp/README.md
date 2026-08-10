# 任务二：MDP、Bellman 方程与动态规划

> 主路线见仓库根 [README](../README.md)。本任务是一个可逐步填充的学习单元：先推导，再在 `src/` 中手写，再用实验解释现象，最后才对照成熟框架。

## 一句话目标

在小型 GridWorld 中用已知转移模型实现 policy evaluation、policy iteration 与 value iteration，亲自验证 Bellman backup。

## 核心概念

MDP 五元组；Bellman expectation/optimality；收缩映射；动态规划

## 本任务交付

手写可配置 GridWorld，打印价值表和贪心策略，并比较两种迭代算法的收敛轮数。

## Definition of Done

- [ ] M1：能计算一个状态的 Bellman expectation backup
- [ ] M2：policy evaluation 收敛
- [ ] M3：policy iteration 找到最优策略
- [ ] M4：value iteration 与其策略一致。

## 建议步骤

1. 在 `notes/` 手写或整理关键公式，标明每一个期望来自何处。
2. 先创建最小的、可复现的环境与随机种子，完成 `src/` 的朴素实现。
3. 增加训练循环和评测循环，输出曲线到 `figures/`、指标到 `notes/`。
4. 做至少一个只改动一个变量的对照实验，并解释结果。
5. 最后阅读对应框架实现或 Stable-Baselines3 / TRL 文档，对照而不替代手写版本。

## 前置知识

任务 1 的期望与采样概念。

## 实现边界与常见提醒

先在 2×2 解析 MDP 手算一轮，再扩到 GridWorld。

## 当前目录状态

本仓库当前只建立教学骨架，尚未提供标准答案或完整 `eval/run.py`。后续按学习进度补充本任务的精确接口、数值单测和训练脚本；现在可运行：

```bash
python eval/run.py
```

它会确认任务骨架已就绪。

## 推荐记录格式

在 `notes/experiment.md` 中记录：日期、Git commit、环境版本、种子、超参、训练步数、最终指标、曲线位置、一个失败现象和你的解释。

