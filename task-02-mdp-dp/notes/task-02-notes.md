# Task 02 学习笔记：MDP、Bellman Equation 与 Dynamic Programming

> 主线案例：寻宝机器人从 Bandit 升级到二维 GridWorld。机器人不再“选一个区域后自动回基地”，而是在地图中一步一步移动，由此引出 State、Transition、Policy、Return、Value Function、Bellman Equation、Policy Iteration 与 Value Iteration。

---

## 1. Task 02 学习目标

Task 02 的目标不是先进入 Q-Learning，而是先把强化学习最核心的数学骨架建立起来：

$$
\text{MDP}
\rightarrow
\pi(a|s)
\rightarrow
G_t
\rightarrow
V_\pi(s)
\rightarrow
Q_\pi(s,a)
\rightarrow
\text{Bellman Equation}
\rightarrow
\text{Policy Evaluation}
\rightarrow
\text{Policy Iteration}
\rightarrow
\text{Value Iteration}
$$

完成本 Task 后，应能够回答：

1. 什么是 MDP？为什么 Bandit 不是完整的序列决策问题？
2. State、Action、Transition、Reward 分别表示什么？
3. 什么叫 Markov Property？
4. Reward 与 Return 有什么区别？
5. \(V_\pi(s)\) 与 \(Q_\pi(s,a)\) 分别表示什么？
6. Bellman Equation 为什么看起来只计算一步，却实际上包含整个未来？
7. Policy Evaluation、Policy Improvement、Policy Iteration 分别做什么？
8. Value Iteration 为什么可以把“评价 + 改进”压缩到同一个 Bellman backup 中？
9. 为什么这些方法属于 model-based Dynamic Programming？

---

# 2. GridWorld：从 Bandit 到 MDP

Task 1 的寻宝机器人更接近：

```text
选择一个区域
    ↓
得到 reward
    ↓
本轮结束
```

Task 2 中机器人真正进入地图：

```text
S  .  .  .
.  #  .  .
.  .  .  .
.  .  .  G
```

- `S`：Start
- `G`：Goal / Treasure
- `#`：Obstacle
- `.`：普通状态

机器人动作：

$$
\mathcal A=
\{\text{UP},\text{DOWN},\text{LEFT},\text{RIGHT}\}
$$

一次交互变成：

$$
S_t
\xrightarrow{A_t}
R_{t+1},S_{t+1}
$$

这就是序列决策问题。

---

# 3. MDP 定义

标准 Markov Decision Process：

$$
\boxed{
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma)
}
$$

在 GridWorld 中：

| MDP 元素 | GridWorld 中的含义 |
|---|---|
| \(\mathcal S\) | 机器人所有合法位置 |
| \(\mathcal A\) | 上、下、左、右 |
| \(P(s'|s,a)\) | 从状态 \(s\) 执行动作 \(a\) 后转移到 \(s'\) 的概率 |
| \(R\) | 状态转移后获得的即时奖励 |
| \(\gamma\) | Discount Factor |

当前环境是 deterministic，因此对给定 \(s,a\)，下一状态唯一。

例如：

$$
s=(2,1),\quad a=\text{RIGHT}
$$

则：

$$
P((2,2)|(2,1),RIGHT)=1
$$

---

# 4. `env.state` 与“正在分析的 state”不是一回事

这是学习过程中第一个容易混淆的点。

假设真实机器人当前：

```python
env.state == (0, 0)
```

调用：

```python
env.step(GridWorld.RIGHT)
```

表示机器人真的从 `(0,0)` 移动到 `(0,1)`，环境状态会发生变化。

但 Dynamic Programming 中可以直接查询：

```python
env.get_next_state((2, 2), GridWorld.RIGHT)
```

这表示：

> 假设机器人位于 `(2,2)`，执行 RIGHT 会发生什么？

它只是查询环境模型，不会把真实 `env.state` 改成 `(2,2)`。

因此：

- `env.state`：机器人当前真实在哪里。
- MDP 中的 `state`：算法当前正在分析的任意状态。

Dynamic Programming 要计算：

$$
V(s),\quad \forall s\in\mathcal S
$$

所以必须遍历整张地图，而不仅仅是当前真实位置。

---

# 5. Markov Property

Markov Property：

$$
P(S_{t+1}|S_t,A_t,S_{t-1},A_{t-1},\ldots)
=
P(S_{t+1}|S_t,A_t)
$$

直观含义：

> 给定当前状态以后，预测未来不再需要额外查看完整历史。

注意这不等于“过去完全不重要”。

更准确的说法是：

> 与未来有关的历史信息，应当已经被当前 State 总结进去。

例如如果地图中存在“钥匙—门”机制，仅使用：

```python
state = (row, col)
```

可能不够，因为同一个位置下，“有钥匙”和“没有钥匙”的未来转移不同。

这时应把状态扩展为：

```python
state = (row, col, has_key)
```

因此，一个好的 State 应当包含做未来预测与决策所需要的信息。

---

# 6. Policy：Agent 如何选择 Action

Policy 记作：

$$
\boxed{\pi(a|s)}
$$

表示：

> 在状态 \(s\) 下选择动作 \(a\) 的概率。

例如 uniform random policy：

$$
\pi(a|s)=0.25
$$

即：

```text
UP      25%
DOWN    25%
LEFT    25%
RIGHT   25%
```

Policy 回答的问题是：

> 我应该怎么行动？

Value Function 回答的问题则是：

> 这种行动方式长期来看有多好？

---

# 7. Reward 与 Return

## 7.1 Reward

Reward 是单步即时奖励：

$$
R_{t+1}
$$

当前 GridWorld 中：

```text
普通移动：-1
进入 Goal：+10
```

Reward 是环境规则的一部分，不是通过 Value Iteration “算出来”的。

---

## 7.2 Return

Return 表示从当前时刻开始未来所有折扣奖励的累计：

$$
\boxed{
G_t=
R_{t+1}
+\gamma R_{t+2}
+\gamma^2R_{t+3}
+\cdots
}
$$

其中下标 \(t\) 表示：

> 从时刻 \(t\) 开始往后看。

例如：

```text
t=0        t=1        t=2        t=3
S0 --A0--> S1 --A1--> S2 --A2--> Goal
       R1=-1      R2=-1      R3=10
```

则：

$$
G_0=R_1+\gamma R_2+\gamma^2R_3
$$

$$
G_1=R_2+\gamma R_3
$$

$$
G_2=R_3
$$

因此：

$$
\boxed{
G_t\text{ 从 }R_{t+1}\text{ 开始累加}
}
$$

---

# 8. Discount Factor \(\gamma\)

$$
0\leq \gamma\leq1
$$

用于控制未来奖励的重要程度。

- \(\gamma=0\)：只关心下一步 reward。
- \(\gamma\) 较小：偏重近期收益。
- \(\gamma\) 较大：未来收益仍然很重要。

Return 具有递归结构：

$$
\boxed{
G_t=R_{t+1}+\gamma G_{t+1}
}
$$

这是 Bellman Equation 的直接基础。

---

# 9. State Value Function \(V_\pi(s)\)

定义：

$$
\boxed{
V_\pi(s)
=
\mathbb E_\pi[G_t|S_t=s]
}
$$

含义：

> 当前处于状态 \(s\)，之后一直按照策略 \(\pi\) 行动，未来 Return 的期望是多少？

因此 \(V_\pi(s)\) 评价的是：

> “这个状态在策略 \(\pi\) 下有多好？”

它不是即时 reward。

同一个状态在不同 policy 下，Value 可以不同：

$$
V_{\pi_1}(s)\neq V_{\pi_2}(s)
$$

---

# 10. Action Value Function \(Q_\pi(s,a)\)

定义：

$$
\boxed{
Q_\pi(s,a)
=
\mathbb E_\pi[G_t|S_t=s,A_t=a]
}
$$

含义：

> 当前在状态 \(s\)，第一步明确执行动作 \(a\)，之后再按照策略 \(\pi\) 行动，这样长期来看有多好？

因此：

- \(V_\pi(s)\)：这个状态整体有多好？
- \(Q_\pi(s,a)\)：在这个状态下执行这个动作有多好？

在 deterministic GridWorld 中：

$$
\boxed{
Q_\pi(s,a)
=
r+\gamma V_\pi(s')
}
$$

这里 \(V_\pi(s')\) 已经包含从下一状态开始的整个未来。

---

# 11. V 与 Q 的关系

对于固定 policy：

$$
\boxed{
V_\pi(s)
=
\sum_a\pi(a|s)Q_\pi(s,a)
}
$$

如果是 uniform random policy：

$$
\pi(a|s)=0.25
$$

则：

$$
V_\pi(s)
=
\frac{
Q_\pi(s,UP)
+Q_\pi(s,DOWN)
+Q_\pi(s,LEFT)
+Q_\pi(s,RIGHT)
}{4}
$$

实验中得到：

```text
Average Q: -3.9772894907
V(s):      -3.9772890096
```

两者几乎相同，是因为：

$$
V_\pi(s)
=
\text{Average Q}
$$

微小误差来自 Policy Evaluation 使用有限收敛阈值，例如：

$$
\theta=10^{-6}
$$

并没有迭代到数学上的无限精度。

---

# 12. 为什么 Bellman 看起来只计算一步，却包含整个未来？

这是本 Task 最关键的理解点之一。

Value 定义本来包含整个未来：

$$
V_\pi(s)
=
\mathbb E_\pi[
R_{t+1}
+\gamma R_{t+2}
+\gamma^2R_{t+3}
+\cdots
]
$$

把后面部分分组：

$$
V_\pi(s)
=
\mathbb E_\pi[
R_{t+1}
+
\gamma(
R_{t+2}
+\gamma R_{t+3}
+\cdots)
]
$$

括号内部就是从下一时刻开始的 Return：

$$
G_{t+1}
$$

而它的期望就是：

$$
V_\pi(S_{t+1})
$$

因此：

$$
\boxed{
V_\pi(s)
=
\mathbb E_\pi[
R_{t+1}
+\gamma V_\pi(S_{t+1})
|S_t=s
]
}
$$

所以 Bellman Equation 虽然计算上只看 one-step transition，但：

$$
V_\pi(s')
$$

已经把后面整个未来压缩进去了。

应牢记：

$$
\boxed{
\text{One-step Bellman backup}
\neq
\text{只考虑一步未来}
}
$$

---

# 13. Bellman Expectation Equation

一般形式：

$$
\boxed{
V_\pi(s)
=
\sum_a
\pi(a|s)
\sum_{s',r}
p(s',r|s,a)
[
r+\gamma V_\pi(s')
]
}
$$

当前 deterministic GridWorld 中可以简化为：

$$
\boxed{
V_\pi(s)
=
\sum_a
\pi(a|s)
[
r+\gamma V_\pi(s')
]
}
$$

含义：

1. Policy 决定不同 action 的概率。
2. 环境决定执行 action 后可能得到的 \(s',r\)。
3. 对所有可能结果取期望。

---

# 14. Iterative Policy Evaluation

目标：

> 给定固定 policy \(\pi\)，计算 \(V_\pi(s)\)。

初始化：

$$
V_0(s)=0
$$

然后不断做 Bellman update：

$$
\boxed{
V_{k+1}(s)
=
\sum_a
\pi(a|s)
[
r+\gamma V_k(s')
]
}
$$

直到：

$$
\max_s
|V_{k+1}(s)-V_k(s)|
<\theta
$$

其中：

$$
\delta_k
=
\max_s
|V_{k+1}(s)-V_k(s)|
$$

代码停止条件：

```python
if delta < theta:
    break
```

---

# 15. 为什么 Random Policy Evaluation 会迭代 111 次？

实验中：

```text
Number of iterations: 111
```

意味着：

> 从 \(V_0(s)=0\) 开始，共进行了 111 次 Bellman sweep，第 111 次更新后满足收敛条件。

如果：

$$
\theta=10^{-6}
$$

那么停止条件为：

$$
\boxed{
\max_s
|V_{112}(s)-V_{111}(s)|
<10^{-6}
}
$$

因此 `Iteration 111` 已经是此次 Policy Evaluation 的收敛结果。

它不是“还没有迭代完成”。

---

# 16. 为什么 Random Policy 收敛以后 Value 还是负的？

Random Policy Evaluation 得到的是：

$$
\boxed{
V_{\pi_{\text{random}}}(s)
}
$$

它回答：

> 如果机器人永远随机乱走，各个状态值多少？

由于机器人会绕路、撞墙，并不断获得：

$$
-1
$$

所以很多状态的长期 Return 可以是负数。

这和是否收敛无关。

应区分：

$$
\boxed{
\text{Policy Evaluation 收敛}
\neq
\text{Policy 已经最优}
}
$$

“收敛”只表示：

> 当前 policy 的 Value 已经算准了。

---

# 17. Policy Improvement

已经获得 \(V_\pi(s)\) 后，可以计算：

$$
Q_\pi(s,a)
=
r+\gamma V_\pi(s')
$$

然后在每个状态选择：

$$
\boxed{
\pi'(s)
=
\arg\max_aQ_\pi(s,a)
}
$$

- `max`：最大 Q-value 是多少。
- `argmax`：哪个 action 取得最大 Q-value。

Policy Improvement 的核心思想：

> 当前策略的状态价值是所有动作 Q-value 的加权平均，而最大 Q-value 不会小于这个加权平均，因此选择最大 Q 的动作不会使策略变差。

---

# 18. Policy Iteration

组合：

$$
\boxed{
\text{Policy Evaluation}
+
\text{Policy Improvement}
}
$$

流程：

```text
初始化 policy π0
      ↓
Policy Evaluation
      ↓
得到 Vπ0
      ↓
Policy Improvement
      ↓
得到 π1
      ↓
重新 Evaluation
      ↓
...
      ↓
Policy 不再变化
      ↓
π*
```

最终得到：

$$
\boxed{\pi^*}
$$

即最大化期望 Return 的最优策略。

需要注意：

> Policy Iteration 不是暴力枚举所有 policy，而是通过“评价当前策略 → 局部贪心改进”不断逼近最优策略。

---

# 19. Bellman Optimality Equation

定义最优 State Value：

$$
\boxed{
V_*(s)=\max_\pi V_\pi(s)
}
$$

对于 deterministic GridWorld：

$$
\boxed{
V_*(s)
=
\max_a
[
r+\gamma V_*(s')
]
}
$$

一般 stochastic MDP：

$$
\boxed{
V_*(s)
=
\max_a
\sum_{s',r}
p(s',r|s,a)
[
r+\gamma V_*(s')
]
}
$$

和 Bellman Expectation Equation 最大的区别：

固定 policy：

$$
V_\pi(s)
=
\sum_a\pi(a|s)Q_\pi(s,a)
$$

最优情况：

$$
V_*(s)
=
\max_aQ_*(s,a)
$$

即：

$$
\boxed{
\sum_a\pi(a|s)
\rightarrow
\max_a
}
$$

---

# 20. Value Iteration

初始化：

$$
V_0(s)=0
$$

然后不断执行：

$$
\boxed{
V_{k+1}(s)
=
\max_a
[
r+\gamma V_k(s')
]
}
$$

直到 Value 收敛。

最后提取：

$$
\boxed{
\pi^*(s)
=
\arg\max_a
[
r+\gamma V_*(s')
]
}
$$

Value Iteration 可以理解为：

> 不再完整评价某个 policy，而是在每次 Value update 中直接执行一次贪心 improvement。

---

# 21. Policy Iteration vs Value Iteration

## Policy Iteration

```text
πk
 ↓
完整/充分 Policy Evaluation
 ↓
Vπk
 ↓
Policy Improvement
 ↓
πk+1
```

特点：

> 先把当前 policy 看清楚，再改策略。

## Value Iteration

```text
V0
 ↓
Optimal Bellman Backup
 ↓
V1
 ↓
Optimal Bellman Backup
 ↓
...
 ↓
V*
 ↓
提取 π*
```

特点：

> 每次更新 Value 时直接选择当前最优动作。

两者最终都求：

$$
V_*,\pi^*
$$

实验中应看到：

$$
V_{PI}\approx V_{VI}
$$

---

# 22. 为什么 PI 与 VI 的最终 Policy 可能不完全相同？

某些状态可能存在多个动作：

$$
Q_*(s,a_1)=Q_*(s,a_2)
$$

例如：

```text
RIGHT → DOWN
```

和：

```text
DOWN → RIGHT
```

都能以同样的 Return 到达 Goal。

因此：

$$
\boxed{
\text{Optimal Value 通常唯一，但 Optimal Policy 可能不唯一}
}
$$

不同实现中的 tie-breaking 可能导致箭头不同，但只要 Value 相同，策略仍然可能都是最优的。

---

# 23. 为什么 Policy Evaluation 的图和 Optimal Value 图不同？

实验中出现：

```text
Random Policy Evaluation - Iteration 111
```

Value 大多为负。

而：

```text
Optimal Value - Policy Iteration
```

Value 为正且明显更高。

原因不是前者“还没收敛”，而是它们评价的是不同 policy：

第一张：

$$
V_{\pi_{\text{random}}}(s)
$$

第二张：

$$
V_*(s)
$$

因此应区分：

```text
Random Policy Evaluation 收敛
        ↓
只是把“随机乱走有多差”算准确

Policy Iteration
        ↓
不断改进 Policy

Optimal Value
        ↓
“采用最优行为时每个状态有多好”
```

---

# 24. Model-based Dynamic Programming

Policy Evaluation、Policy Iteration、Value Iteration 有一个共同前提：

> 已知完整环境模型。

算法可以直接查询：

```python
env.get_next_state(state, action)
env.get_reward(state, action, next_state)
```

也就是已知：

$$
P(s'|s,a)
$$

和：

$$
R(s,a,s')
$$

因此不需要机器人真的把所有状态—动作组合都试一遍。

这就是：

$$
\boxed{
\text{Model-based Dynamic Programming}
}
$$

后续 Q-Learning 会去掉这个前提：

> 如果不知道 \(P\) 和 \(R\)，就通过真实 interaction/sample 学习。

---

# 25. 与 Q-Learning 的联系

Bellman Optimality 的 Q 形式：

$$
\boxed{
Q_*(s,a)
=
\sum_{s',r}
p(s',r|s,a)
[
r+\gamma\max_{a'}Q_*(s',a')
]
}
$$

deterministic 情况下：

$$
Q_*(s,a)
=
r+\gamma\max_{a'}Q_*(s',a')
$$

以后 Q-Learning：

$$
\boxed{
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
[
r+\gamma\max_{a'}Q(s',a')
-Q(s,a)
]
}
$$

其中 target：

$$
r+\gamma\max_{a'}Q(s',a')
$$

直接来自 Bellman Optimality Equation。

因此路线是：

```text
Bellman Optimality Equation
        ↓
Value Iteration
        ↓
不知道完整环境模型
        ↓
从真实 transition 中采样
        ↓
Q-Learning
        ↓
Q 用神经网络逼近
        ↓
DQN
```

---

# 26. 实验结论

## 26.1 Value Propagation

从：

$$
V_0(s)=0
$$

开始，Goal 附近状态最先受到 `+10` reward 的影响。

经过多轮 Bellman backup：

```text
Goal
 ↓
邻近状态
 ↓
更远状态
 ↓
...
```

最终长期未来信息传播到整张地图。

---

## 26.2 PI vs VI

Policy Iteration 与 Value Iteration 最终得到：

$$
V_{PI}\approx V_{VI}\approx V_*
$$

即使策略箭头不完全一致，也可能是由于多个 optimal action 并存。

---

## 26.3 Discount Factor

测试：

$$
\gamma\in\{0,0.5,0.9,0.99\}
$$

可以看到：

- \(\gamma\) 小：未来 Goal reward 对远处状态影响弱。
- \(\gamma\) 大：未来 reward 衰减慢，Goal 对远处状态影响更强。

但不能简单理解为：

> \(\gamma\) 越大永远越好。

它决定的是时间偏好与未来奖励权重。

---

# 27. 高频易混淆问题汇总

## Q1：机器人真实在 `(0,0)`，为什么 Value Iteration 可以直接计算 `(2,2)`？

因为 Dynamic Programming 在使用已知环境模型做假设性推演。

```python
env.step(action)
```

表示真实交互。

```python
env.get_next_state(state, action)
```

表示模型查询。

Value Iteration 要计算所有：

$$
V(s),\quad s\in\mathcal S
$$

因此会遍历整张地图。

---

## Q2：\(G_t\) 的下标 \(t\) 是什么意思？

表示：

> 从时刻 \(t\) 开始往后看的累计 Return。

$$
G_t=R_{t+1}+\gamma R_{t+2}+\cdots
$$

---

## Q3：Value 不是要考虑未来所有状态吗？为什么 Bellman 只看下一状态？

因为：

$$
V(s')
$$

已经代表从下一状态开始整个未来的 Return。

因此：

$$
V(s)=r+\gamma V(s')
$$

是把整个未来递归压缩起来，而不是忽略未来。

---

## Q4：我们是不是在比较不同策略，最后选择奖励最大的策略？

直觉上可以这么理解，但更准确是：

> 寻找能够最大化 expected Return 的策略。

Reward 是环境直接给出的即时反馈。

算法真正迭代计算的是：

$$
V_\pi(s),Q_\pi(s,a)
$$

再用它们不断改进 policy。

---

## Q5：V 和 Q 最本质的区别是什么？

$$
V_\pi(s)
$$

问：

> 我在这里，按照 \(\pi\) 走下去有多好？

$$
Q_\pi(s,a)
$$

问：

> 我在这里，第一步先做 \(a\)，之后按照 \(\pi\) 走下去有多好？

Q 比 V 多指定了第一步 action。

---

## Q6：为什么 Average Q 和 V 几乎一样？

对于 random policy：

$$
\pi(a|s)=0.25
$$

所以：

$$
V_\pi(s)
=
\frac14
\sum_aQ_\pi(s,a)
$$

即 Average Q。

微小数值误差来自迭代收敛阈值。

---

## Q7：为什么 Policy Evaluation 要迭代 111 次？

因为 Bellman Equation 是递归方程，一开始不知道真正 \(V_\pi\)。

从：

$$
V_0(s)=0
$$

不断迭代：

$$
V_0\rightarrow V_1\rightarrow\cdots
$$

直到：

$$
\max_s|V_{k+1}(s)-V_k(s)|<\theta
$$

111 表示该实验中满足收敛条件所需的 Bellman sweep 数量。

---

## Q8：为什么 Iteration 111 的图和 Optimal Value 图不一样？

因为：

$$
\text{Iteration 111}
$$

是 random policy 的收敛价值：

$$
V_{\pi_{\text{random}}}
$$

而 Optimal Value 是：

$$
V_*
$$

“收敛”不等于“最优”。

---

# 28. 核心公式速查

## Return

$$
G_t
=
R_{t+1}
+\gamma R_{t+2}
+\gamma^2R_{t+3}
+\cdots
$$

$$
G_t
=
R_{t+1}
+\gamma G_{t+1}
$$

## State Value

$$
V_\pi(s)
=
\mathbb E_\pi[G_t|S_t=s]
$$

## Action Value

$$
Q_\pi(s,a)
=
\mathbb E_\pi[G_t|S_t=s,A_t=a]
$$

## V-Q Relationship

$$
V_\pi(s)
=
\sum_a\pi(a|s)Q_\pi(s,a)
$$

## Bellman Expectation

$$
V_\pi(s)
=
\sum_a\pi(a|s)
\sum_{s',r}
p(s',r|s,a)
[
r+\gamma V_\pi(s')
]
$$

## Policy Improvement

$$
\pi'(s)
=
\arg\max_aQ_\pi(s,a)
$$

## Bellman Optimality

$$
V_*(s)
=
\max_a
\sum_{s',r}
p(s',r|s,a)
[
r+\gamma V_*(s')
]
$$

## Optimal Q

$$
Q_*(s,a)
=
\sum_{s',r}
p(s',r|s,a)
[
r+\gamma\max_{a'}Q_*(s',a')
]
$$

---

# 29. Task 02 工程结构

```text
task-02/
├── src/
│   ├── __init__.py
│   ├── gridworld.py
│   ├── policy_evaluation.py
│   ├── policy_iteration.py
│   └── value_iteration.py
│
├── notebooks/
│   ├── 01-learning.ipynb
│   └── 02-experiments.ipynb
│
├── notes/
│   └── task-02-notes.md
│
├── figures/
└── README.md
```

职责：

- `gridworld.py`：MDP 环境模型。
- `policy_evaluation.py`：固定 policy 的价值计算。
- `policy_iteration.py`：Evaluation + Improvement。
- `value_iteration.py`：Bellman Optimality backup。
- `01-learning.ipynb`：完整学习主线、公式验证。
- `02-experiments.ipynb`：Value propagation、PI/VI 对比、\(\gamma\) 实验。
- `notes/task-02-notes.md`：完整学习笔记。

---

# 30. Task 02 最终认识

Task 02 最重要的不是记住 Policy Iteration 或 Value Iteration 的代码，而是形成下面这条逻辑链：

```text
环境给即时 Reward
        ↓
Agent 真正关心长期 Return
        ↓
Value Function 预测长期 Return
        ↓
Bellman Equation 把“整个未来”
压缩成“当前一步 + 下一状态价值”
        ↓
给定 Policy 可以 Evaluation
        ↓
利用 Q 可以 Improvement
        ↓
不断改进得到 Optimal Policy
```

最核心的一句话：

$$
\boxed{
\text{Value = Immediate Reward + Discounted Future Value}
}
$$

而 Task 02 与后续真正 model-free RL 的连接点是：

> Dynamic Programming 假设我们知道完整环境模型；后续算法将研究当模型未知时，如何通过 sample 和 interaction 学习相同的 Bellman structure。


---

## Task 02 实验记录

固定 GridWorld 与初始策略，比较 policy evaluation 的收敛轨迹、policy iteration/value iteration 的最终价值，并记录 gamma 改变后 start state value 的变化。

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
