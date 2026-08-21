# Task 01：Multi-Armed Bandit 学习笔记

## 1. Task 目标

Task 1 的目标不是掌握复杂的强化学习算法，而是建立强化学习最基础的几个概念：

- Agent 如何通过与 Environment 交互获得经验；
- Action、Reward、真实价值与估计价值之间的区别；
- Agent 如何根据历史 Reward 更新自己的价值估计；
- 为什么只选择当前最优动作会失败；
- Exploration 与 Exploitation 为什么是强化学习中的基本矛盾；
- ε-Greedy、UCB 和 Thompson Sampling 分别如何处理探索问题。

本任务使用“寻宝机器人”作为统一例子。

机器人每轮从基地出发，在多个藏宝区域中选择一个区域进行探索；探索结束后自动返回基地，因此当前问题没有状态转移，只需要决定“这一次去哪个区域”。

---

## 2. Multi-Armed Bandit

假设共有 \(K\) 个区域：

$$
A_t \in \{0,1,\ldots,K-1\}
$$

机器人每次选择一个区域 \(A_t\)，环境返回一个 Reward：

$$
R_t
$$

在当前 Bernoulli Bandit 中：

$$
R_t \in \{0,1\}
$$

其中：

- \(R_t=1\)：找到宝物；
- \(R_t=0\)：没有找到宝物。

每个区域都有一个固定但对 Agent 隐藏的宝物发现概率。

例如：

$$
q_* = [0.15,0.30,0.45,0.60,0.75]
$$

则 Area 4 的真实价值为：

$$
q_*(4)=0.75
$$

表示每次选择 Area 4：

$$
P(R=1|A=4)=0.75
$$

$$
P(R=0|A=4)=0.25
$$

需要注意，**动作固定并不意味着每次 Reward 固定**。固定的是该动作对应的 Reward Distribution。

---

## 3. \(R_t\)、\(q_*(a)\) 和 \(Q_t(a)\)

这是 Task 1 最需要区分的三个量。

### 3.1 Reward

$$
R_t
$$

表示第 \(t\) 次交互实际获得的奖励。

它是一次随机采样结果。

### 3.2 真实动作价值

$$
q_*(a)
=
\mathbb{E}[R_t|A_t=a]
$$

表示如果长期选择动作 \(a\)，能够获得的平均 Reward。

它属于 Environment 的真实规律，Agent 并不知道。

### 3.3 Agent 的动作价值估计

$$
Q_t(a)
$$

表示 Agent 根据当前已经获得的数据，对 \(q_*(a)\) 的估计。

因此学习过程可以理解为：

$$
Q_t(a)
\rightarrow
q_*(a)
$$

---

## 4. Sample Average 与增量更新

如果动作 \(a\) 已经被选择 \(N(a)\) 次，其 Reward 为：

$$
R_1,R_2,\ldots,R_N
$$

那么最直接的估计是：

$$
Q(a)
=
\frac{1}{N}
\sum_{i=1}^{N}R_i
$$

但没有必要保存所有历史 Reward。

可以使用增量形式：

$$
\boxed{
Q(a)
\leftarrow
Q(a)
+
\frac{1}{N(a)}
[R-Q(a)]
}
$$

其中：

$$
R-Q(a)
$$

表示新观察与当前估计之间的误差。

可以把整个更新统一理解为：

$$
\boxed{
New
=
Old
+
StepSize
\times
Error
}
$$

在当前 Sample Average 中：

$$
\alpha=\frac{1}{N(a)}
$$

所以：

$$
Q(a)\leftarrow Q(a)+\alpha[R-Q(a)]
$$

这个结构在后续 TD、Q-Learning 等算法中还会再次出现。

---

## 5. 一个 Agent 的基本组成

当前 Task 中，一个 Agent 可以简单拆成三个部分：

### Knowledge

```python
Q[a]
N[a]
```

- \(Q(a)\)：对动作价值的估计；
- \(N(a)\)：动作被选择的次数。

### Decision

```python
select_action()
```

决定下一步选择哪个动作。

### Learning

```python
update(action, reward)
```

根据 Reward 更新当前认知。

最核心的交互过程是：

```python
action = agent.select_action()
reward = env.step(action)
agent.update(action, reward)
```

也就是：

$$
Agent
\rightarrow
Action
\rightarrow
Environment
\rightarrow
Reward
\rightarrow
Agent
$$

这就是强化学习最基础的交互闭环。

---

## 6. Greedy

Greedy 的决策规则是：

$$
A_t
=
\arg\max_a Q_t(a)
$$

即：

> 永远选择当前认为价值最高的动作。

它只进行 Exploitation。

Greedy 的问题是，早期 Reward 含有随机性。

例如某个真实价值较低的区域第一次恰好获得：

$$
R=1
$$

那么它的 \(Q(a)\) 可能暂时非常高。

如果 Agent 从此一直选择该区域，就可能永远没有机会发现真正更好的区域。

因此：

$$
\boxed{
价值估计方法正确
\neq
动作选择策略正确
}
$$

---

## 7. Exploration vs Exploitation

### Exploitation

利用已有知识：

> 选择当前看来最好的动作，希望立即获得较高 Reward。

### Exploration

主动获取新信息：

> 尝试当前还不确定的动作，即使它暂时不是当前最优动作。

因此 Bandit 的核心问题是：

$$
\boxed{
Exploration
\quad vs\quad
Exploitation
}
$$

只利用可能过早相信错误判断；

只探索则会一直浪费已经学到的信息。

---

## 8. ε-Greedy

ε-Greedy 是最简单的探索策略。

$$
A_t=
\begin{cases}
\text{random action}, & \text{概率 }\epsilon\\
\arg\max_a Q_t(a), & \text{概率 }1-\epsilon
\end{cases}
$$

例如：

$$
\epsilon=0.1
$$

表示大约：

- 90% 的时间进行 Exploitation；
- 10% 的时间进行 Exploration。

其核心优点是简单。

缺点也很明显：

> Exploration 是随机的，并不知道哪个动作真正值得探索。

即使某个动作已经被证明很差，ε-Greedy 仍然可能随机选择它。

---

## 9. UCB

Upper Confidence Bound 的核心思想是：

$$
\boxed{
当前估计价值
+
不确定性奖励
}
$$

常见形式：

$$
A_t
=
\arg\max_a
\left[
Q_t(a)
+
c
\sqrt{
\frac{\ln t}
{N_t(a)}
}
\right]
$$

其中：

$$
Q_t(a)
$$

表示当前价值估计；

而：

$$
c
\sqrt{
\frac{\ln t}
{N_t(a)}
}
$$

是 Exploration Bonus。

如果：

$$
N(a)
$$

很小，说明这个动作探索得少，Bonus 会较大。

因此 UCB 的探索不是完全随机的，而是：

> 优先选择“当前看起来不错”或者“还不够确定”的动作。

可以把它概括成：

$$
\boxed{
Optimism\ Under\ Uncertainty
}
$$

---

## 10. Thompson Sampling

当前环境的 Reward 为 Bernoulli：

$$
R\in\{0,1\}
$$

因此可以为每个动作的未知成功概率维护 Beta 分布：

$$
p_a
\sim
Beta(\alpha_a,\beta_a)
$$

初始：

$$
\alpha_a=1,\qquad\beta_a=1
$$

找到宝物：

$$
\alpha_a
\leftarrow
\alpha_a+1
$$

没有找到：

$$
\beta_a
\leftarrow
\beta_a+1
$$

动作选择时，从每个动作的后验分布中采样：

$$
\tilde p_a
\sim
Beta(\alpha_a,\beta_a)
$$

然后：

$$
A_t
=
\arg\max_a
\tilde p_a
$$

Thompson Sampling 的核心区别是：

> 它不仅维护“动作大概有多好”，还显式描述“我对这个判断有多确定”。

数据少时分布宽，探索自然更多；

数据多时分布逐渐收窄，行为自然转向 Exploitation。

---

## 11. 三种探索方法的区别

| 方法 | 探索方式 | 核心思想 |
|---|---|---|
| ε-Greedy | 随机探索 | 偶尔不相信当前最优动作 |
| UCB | 基于不确定性探索 | 价值高或探索少的动作更加值得尝试 |
| Thompson Sampling | 基于后验分布探索 | 通过概率分布自然平衡探索与利用 |

Task 1 不需要记住谁“绝对最好”。

真正需要掌握的是：

$$
\boxed{
它们是在用不同的方法解决同一个
Exploration\text{-}Exploitation
问题
}
$$

---

## 12. 实验指标

### Average Reward

进行多次独立实验时：

$$
\bar R_t
=
\frac{1}{M}
\sum_{i=1}^{M}
R_t^{(i)}
$$

表示在第 \(t\) 个 step，多个独立实验平均获得多少 Reward。

它主要衡量：

> 当前策略实际获得 Reward 的能力。

### Cumulative Reward

$$
C_T
=
\sum_{t=1}^{T}
R_t
$$

表示：

> 从实验开始到当前，一共找到了多少次宝物。

### Optimal Action Rate

如果真实最优动作是：

$$
a^*
=
\arg\max_a q_*(a)
$$

那么：

$$
OptimalActionRate_t
=
\frac{1}{M}
\sum_{i=1}^{M}
\mathbf{1}
(
A_t^{(i)}=a^*
)
$$

它衡量的是：

> 有多少 Agent 在当前 step 真正选择了最优动作。

因此：

- Reward 更强调实际收益；
- Optimal Action Rate 更强调决策是否正确。

---

## 13. 为什么需要多个随机种子

Bandit 中同时存在：

- Environment Reward 的随机性；
- Agent Exploration 的随机性。

因此：

$$
\boxed{
单次实验结果
\neq
算法真实表现
}
$$

通常需要进行多次独立实验：

$$
n_{runs}=100,\ 200,\ldots
$$

然后对不同 run 的结果求平均，近似算法的期望性能。

---

## 14. Task 1 的工程结构

本任务推荐使用：

```text
task-01-bandit/
├── src/
│   ├── treasure_bandit.py
│   ├── greedy.py
│   ├── epsilon_greedy.py
│   ├── ucb.py
│   └── thompson_sampling.py
│
├── notebooks/
│   ├── 01-learning.ipynb
│   └── 02-experiments.ipynb
│
├── notes/
│   └── task-01-notes.md
│
└── figures/
```

其中：

- `src/`：算法核心实现；
- `notebooks/`：运行、实验、可视化和消融；
- `notes/`：自己的理论理解与实验总结；
- `figures/`：保存实验结果。

---

## 15. Task 1 最终应该掌握的内容

完成 Task 1 后，应能够解释：

1. 为什么同一个 Action 多次执行会得到不同 Reward；
2. \(R_t\)、\(q_*(a)\)、\(Q_t(a)\) 分别表示什么；
3. 为什么 Sample Average 可以用增量形式更新；
4. 为什么 Greedy 可能长期选择错误动作；
5. Exploration 和 Exploitation 分别解决什么问题；
6. ε-Greedy、UCB 和 Thompson Sampling 的核心差别；
7. 为什么 RL 实验通常需要多个随机种子；
8. Average Reward 和 Optimal Action Rate 分别衡量什么；
9. 一个最小 RL 系统中 Environment、Agent 和 Interaction Loop 分别负责什么。

---

## 16. 从 Task 1 到 Task 2

Task 1 中有一个重要假设：

> 机器人每次探索结束后都会自动返回基地。

因此不存在真正的状态变化：

$$
A_t
\rightarrow
R_t
$$

但如果取消“自动返回基地”，机器人需要在地图中真正移动：

$$
当前位置
\rightarrow
选择动作
\rightarrow
移动到新位置
\rightarrow
获得奖励
\rightarrow
继续决策
$$

这时当前决策会影响未来状态。

于是需要引入：

$$
S_t
$$

并将问题升级为：

$$
\boxed{
S_t
\rightarrow
A_t
\rightarrow
R_{t+1},S_{t+1}
}
$$

这就是下一阶段需要学习的：

$$
\boxed{
Markov\ Decision\ Process\ (MDP)
}
$$

---

## Task 1 Summary

Task 1 可以压缩成一句话：

$$
\boxed{
Agent
通过与未知环境反复交互，
估计不同动作的价值，
并在 Exploration 与 Exploitation 之间进行权衡，
最终学会选择更好的动作。
}
$$

到这里，Multi-Armed Bandit 部分结束。


---

## Task 01 实验记录

比较四种策略时固定 treasure_probs、步数和 seed；至少记录单轨迹与多 run 的奖励、最优动作率、访问次数和一个探索失败现象。

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
