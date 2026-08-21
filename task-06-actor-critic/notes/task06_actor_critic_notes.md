# Task 6: Actor-Critic 学习笔记

## 1. 算法演进路线

强化学习算法的发展核心原因是：随着环境复杂度增加，无法继续维护完整的状态价值表或动作价值表，因此逐渐转向函数逼近。

整体路线：

Q-Learning\
↓\
DQN\
↓\
Policy Gradient (REINFORCE)\
↓\
REINFORCE + Baseline\
↓\
Actor-Critic\
↓\
N-step Actor-Critic\
↓\
GAE\
↓\
PPO

## 2. 从 Q-Learning 到 DQN

Q-Learning 使用 Q 表：

Q(s,a)

当状态空间很大或者连续时，Q 表无法存储。

DQN 使用神经网络拟合：

Q_theta(s,a)

本质变化：

从： 维护一个表

变成：

使用神经网络逼近函数。

## 3. Policy Gradient 回顾

Policy Gradient 直接学习策略：

π(a\|s)

目标：

最大化累计奖励：

J(theta)=E$$R$$

梯度：

∇J(theta)=E$$∇logπ(a\|s)G_t$$

其中 G_t 是未来累计奖励。

REINFORCE：

theta ← theta + alpha \* G_t \* ∇logπ(a_t\|s_t)

问题：

-   使用完整 Episode Return
-   方差大
-   训练不稳定

## 4. Baseline 的作用

引入状态价值：

V(s)

将：

G_t

变成：

G_t - V(s_t)

即：

Advantage:

A(s,a)=Q(s,a)-V(s)

含义：

不是判断动作有没有奖励，而是判断动作是否比平均水平更好。

作用：

降低 Policy Gradient 方差。

# 5. Actor-Critic

Actor：

负责选择动作：

π(a\|s)

Critic：

负责评价状态：

V(s)

训练流程：

state ↓ Actor选择action ↓ 环境返回reward ↓ Critic估计价值 ↓ TD Error ↓
更新Actor

## TD Error

公式：

δ_t = r_t + γV(s\_{t+1}) - V(s_t)

其中：

V(s_t)

是当前状态价值。

V(s\_{t+1})

是下一状态价值。

两者都来自 Critic 网络。

TD Target:

y_t = r_t + γV(s\_{t+1})

TD Error:

δ_t = y_t - V(s_t)

它可以近似 Advantage：

A_t ≈ δ_t

## 关键问题：Critic 最开始不知道价值怎么办？

初始：

V(s)≈0

第一次更新虽然目标不准确，但是：

-   reward 提供真实反馈
-   新数据不断进入
-   梯度下降不断修正参数

类似不断修正一个估计值：

10 → 15 → 20 → ...

这就是 bootstrap。

# 6. One-step Actor-Critic

One-step 使用：

G_t = r_t + γV(s\_{t+1})

每走一步：

-   计算 TD Target
-   更新 Critic
-   使用 TD Error 更新 Actor

优点：

-   更新快
-   数据利用率高

缺点：

-   强依赖 Critic
-   Critic 初期误差会影响 Actor

实验：

One-step Actor-Critic 可以达到接近 500 reward，但存在后期震荡。

# 7. N-step Actor-Critic

为了减少 Critic 的影响，引入多步真实奖励。

5-step:

G_t\^(5)

= r_t + γr\_{t+1} + γ²r\_{t+2} + γ³r\_{t+3} + γ⁴r\_{t+4} + γ⁵V(s\_{t+5})

代码中通过递推实现：

R = V(s\_{t+n})

for reward in reversed(rewards):

    R = reward + gamma * R

## 为什么递推？

因为：

R_t = r_t + γR\_{t+1}

从最后一步开始向前展开。

# 8. Bias-Variance Tradeoff

不同估计方式：

## One-step TD

bootstrap 多：

优点：

-   方差低

缺点：

-   偏差高

## Monte Carlo

等待 Episode 结束：

优点：

-   偏差低

缺点：

-   方差高

## N-step

位于二者之间：

One-step

↓

N-step

↓

Monte Carlo

# 9. 实验结果

实验环境：

CartPole-v1

比较：

1-step Actor-Critic

vs

5-step Actor-Critic

结果：

## One-step

特点：

-   可以学习成功
-   达到较高 reward
-   后期存在震荡

## 5-step

特点：

-   更早达到高 reward
-   超过400 reward速度更快
-   高性能阶段保持更稳定

原因：

5-step 使用更多真实 reward：

降低 Critic 估计误差影响。

# 10. Task 6 总结

本 Task 完成：

-   Policy Gradient
-   REINFORCE
-   Baseline
-   Actor-Critic
-   TD Error
-   Bootstrap
-   One-step Actor-Critic
-   N-step Actor-Critic
-   Bias-Variance Tradeoff

核心理解：

Actor负责：

提出动作。

Critic负责：

评价状态。

TD Error：

连接 Actor 和 Critic。

算法演进：

REINFORCE

↓

Baseline

↓

Actor-Critic

↓

N-step Actor-Critic

下一阶段：

Actor-Critic

↓

GAE

↓

PPO


---

## Task 06 实验记录

固定网络和环境，比较 one-step 与 n-step；若实现 GAE，再以 lambda=0、0.95、1 做消融，分别记录 episode return、actor loss、critic loss 和稳定性。

| 项目 | 记录 |
|---|---|
| 日期 / commit | |
| Python / PyTorch / Gymnasium | |
| 随机种子 | |
| 环境与任务配置 | |
| 训练步数或 episode | |
| 最终指标 | |
| 曲线 / 输出文件 | |
| 失败现象 | |
| 解释与下一步 | |


## 公式渲染约定

本笔记的块级公式统一使用成对的 `$$` 包裹，行内公式使用单个 `$`；不要把公式放在 ```text 代码块中，否则 Markdown 渲染器会按普通文本显示。
