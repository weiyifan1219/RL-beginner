# Task 7：TRPO 与 PPO —— 从 Trust Region 到 Clipped Policy Optimization

## 1. 本任务目标

Task 7 继续沿着 Task 5～Task 6 的策略梯度路线推进：

$$
\text{REINFORCE}
\rightarrow
\text{Actor-Critic}
\rightarrow
\text{TRPO}
\rightarrow
\text{PPO}
$$

本任务重点回答两个问题：

1. 为什么普通 Policy Gradient / Actor-Critic 的策略更新可能不稳定？
2. TRPO 和 PPO 如何限制策略更新幅度，并提高训练稳定性？

本任务最终掌握：

- Policy Gradient 中的策略更新问题
- Probability Ratio
- KL Divergence
- Trust Region
- TRPO
- Natural Gradient
- Fisher / KL Hessian
- Hessian-Vector Product
- Conjugate Gradient
- Backtracking Line Search
- PPO Clipped Objective
- GAE
- Bias / Variance / Bootstrapping
- Rollout Buffer
- On-policy 与 Off-policy 的区别
- TRPO 与 PPO 的工程实现差异

---

# 2. 从 Task 1 到 Task 7 的知识链

## Task 1：Bandit

核心问题：

> 在不知道奖励规律时，如何平衡探索与利用？

学习：

- Greedy
- ε-Greedy
- UCB
- Softmax

没有状态转移，只考虑：

$$
a \rightarrow R
$$

---

## Task 2：MDP / Dynamic Programming

引入：

- State
- Action
- Transition
- Reward
- Policy
- Value Function

核心：

$$
V^\pi(s)
$$

和 Bellman Equation。

已知环境模型：

$$
P(s'|s,a)
$$

通过 Policy Evaluation / Policy Iteration / Value Iteration 求最优策略。

---

## Task 3：Tabular RL

环境模型未知，只通过交互学习。

学习：

- Monte Carlo
- TD Learning
- SARSA
- Q-learning

其中：

### Monte Carlo

使用完整 Return：

$$
G_t
=
R_t+\gamma R_{t+1}+\gamma^2R_{t+2}+\cdots
$$

不进行 bootstrap。

### TD

$$
V(s_t)
\leftarrow
V(s_t)
+
\alpha
[
R_t+\gamma V(s_{t+1})-V(s_t)
]
$$

使用 bootstrap。

### SARSA

On-policy：

$$
Q(s_t,a_t)
\leftarrow
Q+
\alpha
[
R_t+\gamma Q(s_{t+1},a_{t+1})-Q
]
$$

### Q-learning

Off-policy：

$$
Q(s_t,a_t)
\leftarrow
Q+
\alpha
[
R_t+\gamma\max_aQ(s_{t+1},a)-Q
]
$$

---

## Task 4：DQN

将 Q-table 替换为神经网络：

$$
Q_\theta(s,a)
$$

核心工程机制：

- Replay Buffer
- Target Network

DQN 是典型 off-policy 算法。

---

## Task 5：Policy Gradient

不再间接学习 Q 后选动作，而是直接学习：

$$
\pi_\theta(a|s)
$$

REINFORCE：

$$
\nabla_\theta J(\theta)
=
\mathbb E
[
\nabla_\theta
\log\pi_\theta(a_t|s_t)
G_t
]
$$

加入 Baseline：

$$
A_t=G_t-V(s_t)
$$

以降低梯度估计方差。

---

## Task 6：Actor-Critic

Actor：

$$
\pi_\theta(a|s)
$$

Critic：

$$
V_\phi(s)
$$

使用 TD error：

$$
\delta_t
=
R_t+\gamma V(s_{t+1})-V(s_t)
$$

作为 Advantage 的近似：

$$
A_t\approx\delta_t
$$

以及进一步学习 n-step Actor-Critic。

---

## Task 7：TRPO / PPO

普通 Actor-Critic 的核心问题：

> 梯度告诉我们往哪个方向更新，但没有严格控制一次更新后 policy 改变多少。

TRPO：

$$
\text{Actor-Critic}
+
\text{KL Trust Region}
+
\text{Natural Gradient}
$$

PPO：

$$
\text{Actor-Critic}
+
\text{GAE}
+
\text{Probability Ratio}
+
\text{Clip}
$$

---

# 3. 符号统一

本任务中严格区分 reward 和 probability ratio。

环境即时奖励统一记为：

$$
\boxed{R_t}
$$

PPO / TRPO probability ratio 统一记为：

$$
\boxed{
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
}
$$

避免同时使用 \(r_t\) 表示 reward 和 ratio。

---

# 4. 为什么需要限制 Policy Update？

普通 Policy Gradient：

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta J(\theta)
$$

问题是：

$$
\|\theta_{\text{new}}-\theta_{\text{old}}\|
$$

很小，并不意味着：

$$
\pi_{\text{new}}
$$

和：

$$
\pi_{\text{old}}
$$

很接近。

神经网络参数是共享的，一次较大的参数更新可能同时破坏许多状态下已经学好的行为。

例如：

$$
\pi_{\text{old}}(\text{right}|s)=0.55
$$

一次更新后可能直接变成：

$$
0.95
$$

但一个较大的正 Advantage 只说明：

> 当前采样数据表明这个动作比预期好。

并不能保证：

> 这个动作以后应该以 95% 的概率执行。

因此策略更新需要限制。

---

# 5. Probability Ratio

定义：

$$
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\text{old}}(a_t|s_t)
}
$$

含义：

- \(\rho_t=1\)：新旧策略对该动作概率相同
- \(\rho_t>1\)：新策略提高该动作概率
- \(\rho_t<1\)：新策略降低该动作概率

例如：

$$
\pi_{\text{old}}=0.6
$$

$$
\pi_{\text{new}}=0.7
$$

则：

$$
\rho
=
\frac{0.7}{0.6}
=
1.1667
$$

代表动作概率相对提高约 16.67%。

---

## 5.1 为什么代码存 log probability？

理论：

$$
\rho
=
\frac{\pi_{\text{new}}}{\pi_{\text{old}}}
$$

代码通常：

```python
ratio = torch.exp(
    new_log_prob - old_log_prob
)
```

因为：

$$
\log\frac{a}{b}
=
\log a-\log b
$$

因此：

$$
\rho
=
\exp(
\log\pi_{\text{new}}
-
\log\pi_{\text{old}}
)
$$

例如：

$$
\log 0.7-\log 0.6
=
0.1541
$$

指数还原：

$$
e^{0.1541}=1.1667
$$

log probability 数值更稳定，同时 Policy Gradient 本身也天然使用：

$$
\nabla_\theta\log\pi_\theta(a|s)
$$

---

# 6. On-policy 与 Off-policy

关键不是：

> 是否选择最大概率动作。

也不是：

> action 是否等于 policy。

Action 只是 policy 的一次采样：

$$
a\sim\pi(a|s)
$$

判断标准是：

> 产生训练数据的 Behavior Policy 和正在学习的 Target Policy 是否一致。

On-policy：

$$
\mu=\pi
$$

Off-policy：

$$
\mu\neq\pi
$$

典型：

| 算法 | 类型 |
|---|---|
| REINFORCE | On-policy |
| Actor-Critic | On-policy |
| TRPO | On-policy |
| PPO | On-policy |
| SARSA | On-policy |
| Q-learning | Off-policy |
| DQN | Off-policy |

PPO 即使：

$$
P(\text{left})=0.4,\quad
P(\text{right})=0.6
$$

实际 sample 出 `left`，仍然是 on-policy，因为 `left` 是从当前 policy 分布中采样得到的。

---

# 7. Rollout Buffer 与 Replay Buffer

DQN：

```text
Replay Buffer
长期保存历史 transition
随机抽样
反复训练
```

因为 DQN 是 off-policy。

PPO / TRPO：

```text
Rollout Buffer
收集当前策略的一批 transition
使用若干次
丢弃
重新采样
```

因为它们是 on-policy。

PPO Rollout Buffer 通常保存：

```text
state
action
reward
done
value
old_log_prob
```

---

# 8. Bias / Variance / Bootstrapping

## 8.1 Bias：偏差

表示估计量长期平均结果和真实值之间的偏离程度。

简单记忆：

> Bias = 平均来看准不准。

---

## 8.2 Variance：方差

表示重复采样时估计结果波动有多大。

简单记忆：

> Variance = 稳不稳。

需要注意：

PPO 中讨论的 variance 主要指：

- Return estimate variance
- Advantage estimate variance
- Policy gradient estimate variance

不等同于：

> Episode Return 曲线的标准差一定明显更小。

---

## 8.3 Bootstrapping：自举

RL 中的 bootstrap 指：

> 使用当前已有的价值估计，帮助更新另一个价值估计。

例如：

$$
V(s_t)
\leftarrow
R_t+\gamma V(s_{t+1})
$$

这里：

$$
V(s_{t+1})
$$

本身也是估计值。

TD：

- 更多 bootstrap
- variance 较低
- bias 较高

Monte Carlo：

- 不 bootstrap
- variance 较高
- bias 较低

---

# 9. GAE

Generalized Advantage Estimation：

$$
\delta_t
=
R_t+\gamma V(s_{t+1})-V(s_t)
$$

GAE：

$$
A_t^{GAE}
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+\cdots
$$

递归形式：

$$
\boxed{
A_t
=
\delta_t
+
\gamma\lambda A_{t+1}
}
$$

---

## 9.1 λ 的含义

$$
\lambda=0
$$

退化为 1-step TD：

$$
A_t=\delta_t
$$

更多 bootstrap：

- bias 较高
- variance 较低

$$
\lambda\rightarrow1
$$

接近 Monte Carlo Advantage：

$$
G_t-V(s_t)
$$

- bias 较低
- variance 较高

通常 PPO 使用：

$$
\lambda=0.95
$$

作为折中。

---

## 9.2 GAE 工程实现中的重要修正

早期教学版本错误地在 rollout 最后一步统一使用：

```python
next_value = 0.0
```

这是不正确的。

如果 rollout 是因为：

```text
rollout_steps 达到上限
```

而不是 environment terminal，那么：

$$
V(s_T)\neq0
$$

必须进行 bootstrap：

$$
\delta_{T-1}
=
R_{T-1}
+
\gamma V(s_T)
-
V(s_{T-1})
$$

因此标准实现需要额外传入：

```python
last_value
```

只有真实 terminal：

$$
V(s_T)=0
$$

这一修正对：

$$
\lambda=0
$$

尤其重要，因为 TD(0) 最依赖 bootstrap value。

之前基于错误版本得到的 λ 消融结果不作为有效实验结论。

---

# 10. TRPO

TRPO：Trust Region Policy Optimization。

核心目标：

$$
\boxed{
\max_\theta
\mathbb E_t
[
\rho_t(\theta)A_t
]
}
$$

同时满足：

$$
\boxed{
\mathbb E_s
[
D_{KL}
(
\pi_{\text{old}}(\cdot|s)
\|
\pi_\theta(\cdot|s)
)
]
\leq\delta
}
$$

也就是：

> 尽可能提高 surrogate objective，但新策略不能离旧策略太远。

---

# 11. KL Divergence

离散动作：

$$
D_{KL}(P\|Q)
=
\sum_a
P(a)
\log
\frac{P(a)}{Q(a)}
$$

TRPO 中：

$$
P=\pi_{\text{old}}
$$

$$
Q=\pi_{\text{new}}
$$

KL 越小：

> 新旧策略越接近。

TRPO 使用 KL 而不是参数 L2 距离，是因为：

$$
\boxed{
\text{Parameter Space}
\neq
\text{Policy Distribution Space}
}
$$

参数改得少，并不保证策略分布改得少。

---

# 12. Trust Region

TRPO 中：

$$
D_{KL}
\leq\delta
$$

定义一个允许策略搜索的局部区域。

含义：

> 只相信旧策略附近的局部优化结果。

经典教学设置：

$$
\delta\approx0.01
$$

---

# 13. Natural Gradient

普通梯度：

$$
g=\nabla_\theta J(\theta)
$$

只考虑：

> 参数空间中 objective 上升最快的方向。

Natural Gradient：

$$
\boxed{
F^{-1}g
}
$$

其中：

$$
F
$$

为 Fisher Information Matrix，在 TRPO 中对应 KL 的局部二阶结构。

它回答：

> 在 policy distribution 变化有限的情况下，哪个方向能让 objective 提升最多？

因此：

普通 Actor-Critic：

$$
\text{方向}=g
$$

TRPO：

$$
\text{方向}=F^{-1}g
$$

---

# 14. 为什么需要 Conjugate Gradient？

理论需要：

$$
F^{-1}g
$$

但大型神经网络无法显式构造：

$$
F
$$

更不可能直接求逆。

因此将问题写成：

$$
Fx=g
$$

使用 Conjugate Gradient 求近似解：

$$
x\approx F^{-1}g
$$

所以：

```text
Policy Gradient
      ↓
g
      ↓
Conjugate Gradient
      ↓
F⁻¹g
      ↓
Natural Gradient Direction
```

---

# 15. Hessian-Vector Product

Conjugate Gradient 不需要完整 Fisher / Hessian，只需要计算：

$$
Fv
$$

因此 TRPO 使用：

$$
\text{Hessian-Vector Product}
$$

而不是显式构造巨大的二阶矩阵。

---

# 16. Backtracking Line Search

Natural Gradient 给出了方向。

但最终还要确保实际更新满足：

1. Surrogate Objective 提升
2. KL 不超过 max_kl

因此 TRPO 使用 Backtracking Line Search：

```text
完整step
   ↓
检查 KL / objective
   ↓
不满足
   ↓
缩小step
   ↓
再次检查
```

例如：

```text
step
0.8 step
0.64 step
...
```

直到满足约束。

---

# 17. TRPO 一次完整更新

```text
Rollout
   ↓
GAE
   ↓
Surrogate Objective
   ↓
Policy Gradient g
   ↓
KL Hessian / Fisher
   ↓
Conjugate Gradient
   ↓
Natural Gradient F⁻¹g
   ↓
KL决定最大step
   ↓
Backtracking Line Search
   ↓
New Policy
```

Actor 不使用普通 Adam 更新。

Critic 仍然可以使用：

```text
MSE Loss + Adam
```

---

# 18. PPO

TRPO 稳定，但工程实现复杂：

- KL Constraint
- Hessian / Fisher
- HVP
- Conjugate Gradient
- Line Search

PPO 的目标：

> 保留“不要让 policy 一次变化太大”的思想，同时使用普通一阶优化器。

---

# 19. PPO Clipped Objective

PPO：

$$
L^{CLIP}
=
\mathbb E_t
\left[
\min
\left(
\rho_tA_t,
\operatorname{clip}
(
\rho_t,
1-\epsilon,
1+\epsilon
)
A_t
\right)
\right]
$$

典型：

$$
\epsilon=0.2
$$

---

## 19.1 A > 0

好动作：

$$
A_t>0
$$

希望：

$$
\rho_t>1
$$

即提高动作概率。

如果：

$$
\rho_t>1+\epsilon
$$

则 objective 不再奖励继续增加。

---

## 19.2 A < 0

坏动作：

$$
A_t<0
$$

希望：

$$
\rho_t<1
$$

如果：

$$
\rho_t<1-\epsilon
$$

则 objective 不再奖励继续降低。

---

## 19.3 Clip 不会主动修改 Ratio

例如：

$$
A>0,\quad
\rho=0.7
$$

PPO 不会直接把：

$$
\rho
$$

变成：

$$
0.8
$$

真正发生的是：

> PPO 保留该区域的梯度，optimizer 会继续提高该动作的新策略概率。

因此：

$$
\boxed{
\text{Clip 不是把 Ratio 拉回来}
}
$$

而是：

$$
\boxed{
\text{在已经沿正确方向走得太远时停止额外奖励}
}
$$

---

# 20. 为什么会出现 A > 0 但 ρ < 1？

Advantage：

$$
A_t
$$

表示：

> 旧 rollout 数据认为这个 action 比预期好还是差。

Ratio：

$$
\rho_t
$$

表示：

> 当前网络更新后，这个 action 的概率相对旧策略发生了什么变化。

由于：

- 网络参数共享
- 一个 batch 中存在多个状态
- 多个样本梯度可能冲突
- PPO 会对同一批 rollout 更新多个 epoch

所以可能出现：

$$
A_t>0
$$

但：

$$
\rho_t<1
$$

这并不矛盾。

---

# 21. TRPO 与 PPO 的真正关系

非常重要：

$$
\boxed{
\text{Ratio 不是 PPO 发明的}
}
$$

TRPO 已经使用：

$$
\rho_tA_t
$$

构造 surrogate objective。

TRPO：

```text
ρA
+
KL硬约束
```

PPO：

```text
直接把ρA修改为Clipped Objective
```

因此 PPO 的核心简化是：

> 使用 ratio + clip + 一阶优化，近似实现 TRPO 的 trust-region 思想。

演进：

```text
Policy Gradient
      ↓
ρA surrogate objective
      ↓
策略可能更新过猛
      ↓
TRPO
ρA + KL Constraint
      ↓
稳定，但实现复杂
      ↓
PPO
Ratio + Clip + Adam
```

---

# 22. TRPO vs PPO

| 项目 | TRPO | PPO |
|---|---|---|
| 类型 | On-policy | On-policy |
| Actor-Critic | 是 | 是 |
| GAE | 通常使用 | 通常使用 |
| Probability Ratio | 是 | 是 |
| KL Constraint | 硬约束 | 通常无硬约束 |
| Natural Gradient | 是 | 否 |
| Fisher / Hessian | 是 | 否 |
| Conjugate Gradient | 是 | 否 |
| Line Search | 是 | 否 |
| Adam 更新 Actor | 否 | 是 |
| 实现复杂度 | 高 | 低 |
| 工程使用 | 较少 | 非常广泛 |

一句话：

$$
\boxed{
PPO 不一定比 TRPO 理论上更强，
但它以更简单的工程实现获得了类似的稳定更新思想。
}
$$

---

# 23. PPO 的训练单位

需要区分：

## Episode

一次完整任务：

```text
reset
↓
连续交互
↓
terminated / truncated
```

## Environment Step

一次：

$$
(s_t,a_t,R_t,s_{t+1})
$$

## PPO Update

例如：

```text
收集2048 steps
↓
计算GAE
↓
PPO更新若干epoch
```

因此 PPO：

> 训练主要按 rollout/update 组织。

但评价：

> 仍然通常使用 episode return。

当前教学实现只有一个环境，因此 episode 是顺序产生的，不是并行环境。

---

# 24. Task 7 实验

## Experiment 1：PPO on CartPole

观察到：

- 前期 Return 较低
- 随着 rollout/update 增加逐渐提升
- 最终可以达到 CartPole-v1 的高回报区域

验证 PPO baseline 正常工作。

---

## Experiment 2：Actor-Critic vs PPO

实验结果：

```text
Actor-Critic:
281.27 ± 98.19

PPO:
392.05 ± 96.66
```

结论：

- PPO 最终平均性能更高
- PPO 收敛速度整体更快
- Episode Return 的标准差并没有明显下降

重要修正：

> PPO 所谓“降低 variance / 提高稳定性”，主要针对 Advantage / Gradient / Policy Update，不意味着 episode return 的方差一定明显下降。

---

## Experiment 3：PPO Clip ε Ablation

实验：

$$
\epsilon
\in
\{
0.05,
0.2,
0.5
\}
$$

结果：

```text
epsilon=0.05:
362.91 ± 116.70

epsilon=0.2:
217.28 ± 53.23

epsilon=0.5:
216.37 ± 158.98
```

观察：

### ε = 0.05

- 学习明显更慢
- 后期相对平稳

说明：

$$
\epsilon\downarrow
$$

意味着 policy update 更保守。

### ε = 0.5

- 前期可能快速提升
- 后期容易出现明显震荡甚至性能下降

说明：

$$
\epsilon\uparrow
$$

约束减弱，更新更激进。

单一随机种子下最终 reward 排序不能作为普适结论。

真正验证的是：

$$
\boxed{
\epsilon
控制策略更新的保守程度
}
$$

---

## Experiment 4：GAE λ Ablation

最初实验得到异常结果：

- λ=0 学习极慢
- λ=1 反而最快

随后检查发现 GAE 实现存在 rollout boundary bootstrap 问题：

```python
if t == T - 1:
    next_value = 0
```

这会错误地将非 terminal rollout 截断视作真正 terminal。

修正为：

```python
last_value = V(s_T)
```

只有真实 terminal 时：

$$
V(s_T)=0
$$

因此早期 λ 消融结果判定为：

$$
\boxed{
\text{无效实验结果，不用于理论结论}
}
$$

正确理论仍然是：

$$
\lambda=0
\rightarrow
\text{more bootstrap}
\rightarrow
\text{higher bias / lower variance}
$$

$$
\lambda\rightarrow1
\rightarrow
\text{more Monte Carlo}
\rightarrow
\text{lower bias / higher variance}
$$

---

## Experiment 5：TRPO on CartPole

结果：

```text
Final 100 Episodes:
410.19 ± 104.44

Mean KL:
0.006498

Max KL:
0.009551

Line Search Acceptance Rate:
100.00%
```

设置：

$$
\text{max\_kl}=0.01
$$

观察：

$$
\max D_{KL}=0.009551<0.01
$$

验证：

$$
\boxed{
TRPO 的 Trust Region 约束正常工作
}
$$

Return 曲线也成功学习至约 400+。

---

## Experiment 6：TRPO vs PPO

实验采用固定 environment step budget 对齐：

$$
100\times1024
=
102400
$$

environment steps。

这样比单纯按 episode 对齐更公平。

比较：

1. Return vs Environment Steps
2. Policy KL per Update
3. Final 100 Episode Performance

核心预期：

TRPO：

$$
D_{KL}\lesssim0.01
$$

因为存在 KL hard constraint + line search。

PPO：

KL 不保证严格小于 0.01，因为 clip 是 surrogate objective 中的软限制。

该实验的核心不是证明：

$$
PPO>TRPO
$$

而是理解：

> PPO 以显著更简单的一阶优化流程近似获得 TRPO 的稳定 policy update 思想。

---

# 25. 本任务关键问题记录

## Q1：为什么控制策略更新幅度？不会学习更慢吗？

会。

单次更新可能更慢。

但 RL 数据带有采样噪声，Advantage 也是估计值，一次大更新可能破坏原本已经学好的策略。

因此需要：

$$
\text{Learning Speed}
\leftrightarrow
\text{Stability}
$$

的折中。

---

## Q2：A > 0 为什么可能出现 ρ < 1？

因为：

- A 来自旧 rollout
- ρ 描述当前网络更新后的变化
- 网络参数共享
- 不同样本梯度会冲突

因此两者不要求永远同方向。

---

## Q3：A > 0 且 ρ < 1 时，clip 会自动让 ρ 增大吗？

不会。

clip 不直接修改 ratio。

因为此时 PPO objective 仍保留：

$$
\rho A
$$

的梯度。

optimizer 通过提高：

$$
\log\pi_{\text{new}}(a|s)
$$

间接让：

$$
\rho\uparrow
$$

---

## Q4：PPO 为什么是 on-policy？动作不一定选择最大概率动作。

On-policy 和是否 greedy 无关。

只要 action 是：

$$
a\sim\pi_\theta(a|s)
$$

由当前策略采样产生，并用于更新当前策略，就是 on-policy。

---

## Q5：Episode、Update、Epoch 有什么区别？

Episode：

> 一次完整环境任务。

PPO Update：

> 收集一个 rollout 后的一次策略更新周期。

PPO 内部可能对同一 rollout 做多个：

```text
update_epochs
```

它比较像深度学习中对当前 batch 重复优化，但不能简单把 RL episode 等同于 epoch。

---

## Q6：PPO 降低 variance，为什么 Return 标准差没明显降低？

因为这里讨论的 variance 主要是：

- Advantage variance
- Gradient variance
- Policy update instability

不是：

$$
Var(\text{Episode Return})
$$

Episode Return 仍受到：

- 环境初始状态
- stochastic policy sampling
- trajectory randomness

影响。

---

## Q7：TRPO 已经有 Ratio，为什么还需要 PPO？

Ratio 本身只是：

$$
\frac{\pi_{\text{new}}}{\pi_{\text{old}}}
$$

并不会限制更新。

TRPO：

$$
\rho A
+
KL\ Constraint
$$

PPO：

直接将：

$$
\rho A
$$

改造成 clipped objective。

因此 PPO 真正的简化不是发明 ratio，而是：

$$
\boxed{
\text{用 Clip 代替复杂的 Trust-Region 二阶优化}
}
$$

---

## Q8：Natural Gradient 是什么？

普通 Gradient：

> 在参数空间中找到 objective 上升最快的方向。

Natural Gradient：

> 考虑 policy distribution 对参数变化的敏感度后重新修正梯度。

$$
\boxed{
\tilde g=F^{-1}g
}
$$

TRPO 用 Conjugate Gradient 近似求：

$$
F^{-1}g
$$

---

# 26. 最终知识链

Task 7 最核心的演进关系：

```text
REINFORCE
    ↓
直接Policy Gradient
Variance较大

Actor-Critic
    ↓
Critic + Bootstrap
更高样本效率

问题：
Policy update可能过大
    ↓

TRPO
    ↓
Probability Ratio
+
KL Trust Region
+
Natural Gradient
+
Conjugate Gradient
+
Line Search

问题：
实现复杂
    ↓

PPO
    ↓
Probability Ratio
+
Clipped Objective
+
GAE
+
Adam

结果：
保留TRPO核心思想
但工程实现显著简化
```

一句话总结：

$$
\boxed{
TRPO 解决“如何在 Trust Region 内稳定更新策略”，
PPO 则解决“如何用简单的一阶优化近似实现这种稳定更新”。
}
$$

---

# 27. Task 7 工程结构

```text
task-07/
├── src/
│   ├── network.py
│   ├── buffer.py
│   ├── advantage.py
│   ├── trpo.py
│   ├── ppo.py
│   └── ac_baseline.py
│
├── notebooks/
│   ├── 01-learning.ipynb
│   └── 02-experiments.ipynb
│
├── figures/
│
└── notes/
    └── task07_trpo_ppo.md
```

Notebook 约定：

```text
01-learning.ipynb
核心机制理解，不做过多实验

02-experiments.ipynb
一个实验一个主要Cell
完整可运行
```

Notebook 从 sibling `src/` 导入时统一：

```python
import sys
from pathlib import Path

sys.path.append(
    str(Path("../src").resolve())
)
```

然后：

```python
from network import Actor, Critic
from ppo import PPOAgent
```

不默认使用：

```python
from src.xxx import xxx
```

避免 notebook 工作目录导致 import 失败。

---

# 28. Task 7 收尾

Task 7 完成后，应能够回答：

1. 为什么 Policy Gradient 需要控制 policy update？
2. Probability Ratio 的作用是什么？
3. TRPO 为什么使用 KL？
4. Trust Region 是什么？
5. Natural Gradient 和普通 Gradient 有什么区别？
6. 为什么需要 Conjugate Gradient？
7. PPO 为什么可以看作 TRPO 的工程简化？
8. PPO Clip 为什么使用 `min`？
9. GAE 如何在 TD 与 MC 之间调节 bias-variance？
10. 为什么 PPO 是 on-policy，却可以对同一 rollout 更新多个 epoch？
11. Rollout Buffer 和 Replay Buffer 有什么区别？
12. Episode、Environment Step、Policy Update 有什么区别？
13. 为什么 rollout 边界需要正确处理 `last_value` bootstrap？

Task 7 的核心不是记住两个公式，而是理解：

$$
\boxed{
\text{Policy Optimization 的关键问题之一，
不是“能不能更新”，
而是“每次应该更新多大”。
}
$$


---

## Task 07 实验记录

固定 rollout budget，比较 PPO clip、clip epsilon、GAE lambda 与 Actor-Critic baseline；记录 clip fraction、approximate KL、entropy、value loss 和评估回报。

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
