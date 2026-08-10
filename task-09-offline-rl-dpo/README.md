# 任务九：Offline RL、RLHF 与 DPO

> 主路线见仓库根 [README](../README.md)。本任务是一个可逐步填充的学习单元：先推导，再在 `src/` 中手写，再用实验解释现象，最后才对照成熟框架。

## 一句话目标

从固定 transition 数据学习策略，并将“偏好数据上的策略优化”连接到 DPO。此任务优先复用现有 Qwen 或本地 API，不重新下载大模型。

## 核心概念

distribution shift；behavior policy；conservative value；偏好对；reference policy；DPO

## 本任务交付

先做 toy offline dataset 的行为克隆与保守价值学习概念实验；再在一个极小偏好集上实现可单测的 DPO loss，最后可选接入已有 Qwen。

## Definition of Done

- [ ] M1：离线数据 schema 与 OOD action 诊断
- [ ] M2：BC baseline
- [ ] M3：CQL/IQL 至少理解并实现一个 toy 版本
- [ ] M4：纯张量 DPO loss 单测
- [ ] M5：可选用本地 Qwen/服务跑小规模 LoRA DPO。

## 建议步骤

1. 在 `notes/` 手写或整理关键公式，标明每一个期望来自何处。
2. 先创建最小的、可复现的环境与随机种子，完成 `src/` 的朴素实现。
3. 增加训练循环和评测循环，输出曲线到 `figures/`、指标到 `notes/`。
4. 做至少一个只改动一个变量的对照实验，并解释结果。
5. 最后阅读对应框架实现或 Stable-Baselines3 / TRL 文档，对照而不替代手写版本。

## 前置知识

任务 5–8；可复用 `../llm-beginner` 的模型和服务。

## 实现边界与常见提醒

模型路径优先级：RL_BEGINNER_MODEL_PATH → LLM_BEGINNER_MODEL_PATH → ../llm-beginner；缺模型时 DPO 张量实验仍可完整完成。

## 当前目录状态

本仓库当前只建立教学骨架，尚未提供标准答案或完整 `eval/run.py`。后续按学习进度补充本任务的精确接口、数值单测和训练脚本；现在可运行：

```bash
python eval/run.py
```

它会确认任务骨架已就绪。

## 推荐记录格式

在 `notes/experiment.md` 中记录：日期、Git commit、环境版本、种子、超参、训练步数、最终指标、曲线位置、一个失败现象和你的解释。

