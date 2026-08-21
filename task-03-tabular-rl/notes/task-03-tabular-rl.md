# Task 03 — Tabular Reinforcement Learning

## 1. 本任务目标

Task 03 从 Task 02 的 Dynamic Programming 进入 **Model-Free Reinforcement Learning**。

Task 02 中环境模型已知：

$$
P(s'|s,a),\qquad R(s,a,s')
$$

因此可以直接通过 Bellman equation 进行策略评估和策略改进。

Task 03 中不再假设知道环境模型，只允许智能体通过：

$$
(S_t,A_t,R_{t+1},S_{t+1})
$$

这样的实际交互样本进行学习。

本任务核心算法：

- Monte Carlo Prediction
- TD(0) Prediction
- SARSA
- Q-Learning

整体主线：

$$
\boxed{DP\rightarrow MC\rightarrow TD\rightarrow SARSA/Q\text{-Learning}}
$$

---

## 2. Task 02 与 Task 03 的核心区别

### Task 02：已知环境模型

DP 可以直接读取：

$$
P(s'|s,a)
$$

并计算所有可能后继状态的期望：

$$
V_\pi(s)
=
\sum_a\pi(a|s)
\sum_{s'}P(s'|s,a)
\left[R(s,a,s')+\gamma V_\pi(s')\right]
$$

因此本质是：

> 已知环境，通过数学期望计算价值。

### Task 03：未知环境模型

不知道：

$$
P(s'|s,a)
$$

只能真正执行动作：

```python
next_state, reward, done = env.step(action)
```

通过大量真实经验逼近价值。

因此：

$$
\boxed{\text{DP：Expectation}}
$$

$$
\boxed{\text{Model-Free RL：Sampling}}
$$

注意：这里所谓“未知环境”并不意味着必须显式学习一个转移模型。MC、TD、SARSA、Q-Learning 都可以直接学习价值函数，而不需要先估计 \(P(s'|s,a)\)。

---

# Part I — Model-Free Prediction

## 3. Monte Carlo Prediction

### 3.1 目标

给定一个固定策略 \(\pi\)，估计：

$$
V_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

其中 Return：

$$
G_t
=
R_{t+1}
+\gamma R_{t+2}
+\gamma^2R_{t+3}
+\cdots
$$

MC 的核心思想：

> 不知道期望，就通过大量完整 episode 得到 Return 样本，再求平均。

即：

$$
V_\pi(s)
\approx
\frac{1}{N}\sum_{i=1}^{N}G^{(i)}
$$

### 3.2 为什么固定策略下 Return 仍然不同？

一个容易混淆的问题是：

> 策略 \(\pi\) 不更新，为什么同一个状态每次得到的 Return 还可能不同？

因为：

$$
\boxed{\text{固定策略}\neq\text{固定轨迹}}
$$

随机性可能来自：

1. 策略本身是 stochastic policy；
2. 环境转移存在随机性。

因此即使策略参数不变，每次 episode 仍可能产生不同轨迹和 Return。

只有当：

- policy deterministic；
- environment deterministic；
- 起始状态完全相同；

轨迹和 Return 才会完全一样。

### 3.3 First-Visit MC 与 Every-Visit MC

如果一条 episode 中某个状态重复出现，例如：

```text
S0 → S1 → S2 → S1 → Terminal
```

则同一个状态对应不同时间点的 Return。

First-Visit MC：

> 每条 episode 中，每个状态只使用第一次访问时的 Return。

Every-Visit MC：

> 每次访问状态都使用对应 Return 更新。

虽然有限样本下结果不同，但二者都在估计：

$$
V_\pi(s)=\mathbb E[G_t\mid S_t=s]
$$

只要状态满足 Markov property，每次重新处于相同状态 \(s\) 时，未来 Return 的条件期望相同，因此随着样本增加，两种方法都可以逼近同一个 \(V_\pi(s)\)。

### 3.4 增量均值更新

无需保存过去所有 Return，可使用：

$$\
V(s)
\leftarrow
V(s)
+
\frac{1}{N(s)}
\left[G-V(s)\right]
$$

更一般写成：

$$
\boxed{
V(s)
\leftarrow
V(s)+\alpha[\text{Target}-V(s)]
}
$$

MC 中：

$$
\boxed{\text{Target}=G_t}
$$

### 3.5 MC 的关键限制

MC 必须等待完整 episode 结束，才能得到完整 Return：

$$
G_t
$$

因此它不适合特别长、甚至没有明确终止状态的任务。

---

## 4. TD(0) Prediction

TD(0) 同样是在固定策略 \(\pi\) 下估计：

$$
V_\pi(s)
$$

但不等待完整 episode。

更新公式：

$$
\boxed{
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
R_{t+1}
+\gamma V(S_{t+1})
-V(S_t)
\right]
}
$$

TD Target：

$$
\boxed{
R_{t+1}+\gamma V(S_{t+1})
}
$$

TD Error：

$$
\boxed{
\delta_t
=
R_{t+1}
+\gamma V(S_{t+1})
-V(S_t)
}
$$

因此：

$$
V(S_t)\leftarrow V(S_t)+\alpha\delta_t
$$

### 4.1 Bootstrap 是什么？

Bootstrap 可以理解为：

> 使用自己当前已有的价值估计，帮助更新另一个价值估计。

TD 使用：

$$
V(S_{t+1})
$$

来更新：

$$
V(S_t)
$$

但 \(V(S_{t+1})\) 自己也只是当前估计值，并不是真实答案。

因此：

$$
\boxed{\text{TD 使用 Bootstrap}}
$$

而 MC 使用完整真实采样 Return：

$$
G_t
$$

所以：

$$
\boxed{\text{MC 不使用 Bootstrap}}
$$

### 4.2 DP / MC / TD 的关系

| 方法 | Sampling | Bootstrap | 是否需要完整模型 |
|---|---|---|---|
| DP | 否 | 是 | 是 |
| MC | 是 | 否 | 否 |
| TD | 是 | 是 | 否 |

可以理解为：

- DP：知道模型，直接计算期望；
- MC：不知道模型，完整跑完后学习；
- TD：不知道模型，走一步就可以学习。

### 4.3 为什么 TD 对训练次数和学习率更敏感？

TD Target 中包含：

$$
V(S_{t+1})
$$

而这个值本身也在学习，因此 Target 是动态变化的。

价值信息需要从接近 Terminal 的状态逐步向前传播：

```text
Terminal
   ↑
  S3
   ↑
  S2
   ↑
  S1
   ↑
  S0
```

因此：

- \(\alpha\) 太大：传播快，但波动更明显；
- \(\alpha\) 太小：更稳定，但需要更多 episode；
- 训练 episode 数不足时，远离 Terminal 的状态可能尚未充分学习。

实验中可以看到，增加训练轮数后 TD 会逐渐逼近 DP reference。

---

## 5. Prediction 阶段总结

MC 与 TD 的统一形式：

$$
\boxed{
\text{Estimate}
\leftarrow
\text{Estimate}
+
\alpha
(\text{Target}-\text{Estimate})
}
$$

区别只在 Target：

$$
\boxed{\text{MC Target}=G_t}
$$

$$
\boxed{\text{TD Target}=R_{t+1}+\gamma V(S_{t+1})}
$$

这一阶段：

$$
\boxed{\pi\text{ 固定，只做 Policy Evaluation}}
$$

---

# Part II — Model-Free Control

## 6. 为什么从 \(V(s)\) 转向 \(Q(s,a)\)？

\(V(s)\) 只能告诉我们：

> 当前策略下，这个状态整体值多少钱？

但 Control 需要解决：

> 当前状态下应该选择哪个动作？

因此需要动作价值：

$$
Q_\pi(s,a)
=
\mathbb E_\pi[G_t\mid S_t=s,A_t=a]
$$

然后可以比较不同 action：

$$
\pi(s)=\arg\max_aQ(s,a)
$$

在 Task 02 中，因为知道环境模型，可以通过：

$$
Q(s,a)
=
\sum_{s'}P(s'|s,a)
[R+\gamma V(s')]
$$

由 \(V\) 推出动作价值。

但 Model-Free Control 不知道 \(P(s'|s,a)\)，因此更自然的做法是直接学习：

$$
\boxed{Q(s,a)}
$$

---

## 7. Exploration vs Exploitation

训练初期：

$$
Q(s,a)=0
$$

此时当前最大的 Q 并不意味着真实动作最优，因为 Q 只是当前有限经验下的估计。

如果永远只选择：

$$
\arg\max_aQ(s,a)
$$

就可能由于早期偶然结果而锁死在一个次优动作上。

因此使用 \(\varepsilon\)-greedy：

$$
A=
\begin{cases}
\text{random action}, & \varepsilon\\
\arg\max_aQ(s,a), & 1-\varepsilon
\end{cases}
$$

其中：

- Exploration：尝试其他动作；
- Exploitation：利用当前认为最优的动作。

### 7.1 为什么训练初期行为近似随机？

初始时所有 Q 相等：

```text
UP     0
DOWN   0
LEFT   0
RIGHT  0
```

即使进入 greedy 分支，多个动作也并列最大，因此 tie-breaking 会随机选择。

所以训练初期整体行为近似 random policy。

---

## 8. Control 中策略到底在哪里更新？

进入 SARSA / Q-Learning 后，已经不再是固定策略 Prediction，而是在做 Control。

代码中通常没有显式写：

```python
policy = improve_policy(...)
```

策略是由当前 Q-table 隐式决定：

$$
\boxed{\pi=\varepsilon\text{-greedy}(Q)}
$$

因此：

```text
更新 Q
  ↓
最新 Q 改变
  ↓
ε-greedy 的动作选择发生变化
  ↓
策略被隐式更新
```

完整闭环：

$$
\boxed{
Q
\rightarrow
\pi
\rightarrow
Experience
\rightarrow
Q
\rightarrow
\pi
\rightarrow\cdots
}
$$

因此 Control 阶段实际上把 Policy Evaluation 与 Policy Improvement 交织在每一步更新中。

---

## 9. SARSA

SARSA 更新：

$$
\boxed{
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+\gamma Q(S_{t+1},A_{t+1})
-Q(S_t,A_t)
\right]
}
$$

名字来自：

$$
S_t,A_t,R_{t+1},S_{t+1},A_{t+1}
$$

即：

$$
\boxed{S-A-R-S-A}
$$

SARSA 使用下一步实际由当前策略选出的：

$$
A_{t+1}
$$

进行更新。

### 9.1 为什么 SARSA 是 On-policy？

行为策略：

$$
\varepsilon\text{-greedy}
$$

Target 中使用的下一动作也由同一个 \(\varepsilon\)-greedy 策略生成：

$$
A_{t+1}\sim\pi(\cdot|S_{t+1})
$$

因此：

$$
\boxed{
\text{Behavior Policy}
=
\text{Target Policy}
}
$$

即 On-policy。

可以直观理解为：

> SARSA 按照“我实际上会怎么走”来学习。

---

## 10. Q-Learning

Q-Learning 更新：

$$
\boxed{
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+\gamma\max_{a'}Q(S_{t+1},a')
-Q(S_t,A_t)
\right]
}
$$

实际行为仍然可以使用：

$$
\varepsilon\text{-greedy}
$$

但更新 Target 始终假设未来选择当前最优动作：

$$
\max_{a'}Q(S_{t+1},a')
$$

### 10.1 为什么 Q-Learning 是 Off-policy？

行为策略：

$$
\varepsilon\text{-greedy}
$$

学习目标对应的策略：

$$
\text{greedy}
$$

因此：

$$
\boxed{
\text{Behavior Policy}
\neq
\text{Target Policy}
}
$$

即 Off-policy。

可以直观理解为：

> Q-Learning 可以一边用带探索的策略采样，一边学习“如果以后都选最优动作”的价值。

---

## 11. SARSA 与 Q-Learning 的核心区别

SARSA Target：

$$
\boxed{R+\gamma Q(S',A')}
$$

Q-Learning Target：

$$
\boxed{R+\gamma\max_{a'}Q(S',a')}
$$

因此：

| 方法 | Behavior Policy | Target Policy | 类型 |
|---|---|---|---|
| SARSA | \(\varepsilon\)-greedy | \(\varepsilon\)-greedy | On-policy |
| Q-Learning | \(\varepsilon\)-greedy | greedy | Off-policy |

---

## 12. 为什么某些 SARSA 状态的 greedy 箭头看起来不是最优？

实验中可能出现：

- 从起点 rollout 已经是最短路径；
- 但某些非主路径状态的 greedy action 仍然看起来不合理。

这不一定意味着算法错误。

原因包括：

1. 从固定起点训练时，不同状态的访问频率不一致；
2. 某些状态很少被访问，Q-value 尚未充分估计；
3. 固定 \(\varepsilon>0\) 时，SARSA 学习的是带探索行为的策略价值，而不是始终纯 greedy 的策略。

因此判断最终策略时，不应只看某一个不常访问状态，而应同时观察：

- greedy rollout 是否能到达 Goal；
- 是否接近最短路径；
- episode return 是否稳定提升。

---

# 13. Task 03 最核心统一公式

Task 03 中几个算法看起来不同，但都可以写成：

$$
\boxed{
\text{Estimate}
\leftarrow
\text{Estimate}
+
\alpha
(\text{Target}-\text{Estimate})
}
$$

不同算法仅仅更换 Target：

$$
\boxed{
\begin{aligned}
MC &: G_t\$$2mm]
TD &: R_{t+1}+\gamma V(S_{t+1})\$$2mm]
SARSA &: R_{t+1}+\gamma Q(S_{t+1},A_{t+1})\$$2mm]
Q\text{-Learning} &: R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')
\end{aligned}
}
$$

这是本任务最重要的统一视角。

---

# 14. Task 03 知识结构

```text
                     Tabular RL
                         │
          ┌──────────────┴──────────────┐
          │                             │
      Prediction                    Control
          │                             │
     固定策略 π                    策略不断变化
          │                             │
      学习 Vπ(s)                   学习 Q(s,a)
          │                             │
     ┌────┴────┐                 ┌──────┴──────┐
     │         │                 │             │
    MC        TD               SARSA       Q-Learning
     │         │                 │             │
完整 Return  Bootstrap        On-policy     Off-policy
```

---

# 15. 从 Task 03 到 Task 04

Tabular Q-Learning 的核心是维护 Q-table：

$$
Q(s,a)
$$

但当状态空间变得很大时，例如图像输入、复杂地图或连续高维观测，Q-table 无法为每一个状态动作对单独存储数值。

Task 04 将解决：

> 如果 Q-table 放不下，能不能用神经网络近似 \(Q(s,a)\)？

即：

$$
\boxed{
Q\text{-Learning}
\rightarrow
Deep\ Q\text{-Network (DQN)}
}
$$

DQN 本质上仍然沿用 Task 03 的 Q-Learning Target：

$$
R_{t+1}
+
\gamma
\max_{a'}Q(S_{t+1},a')
$$

最大的变化是：

$$
\boxed{
Q\text{-table}
\rightarrow
Q_\theta(s,a)
}
$$

也就是使用神经网络参数 \(\theta\) 来近似动作价值函数。


---

## Task 03 实验记录

在同一 GridWorld 和随机策略下比较 DP、MC、TD(0) 的 RMSE；再固定 epsilon、alpha 和 episode 数比较 SARSA/Q-Learning 的回报与最终策略。

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
