# Task 04 — Function Approximation & Deep Q-Network (DQN)

> 本任务目标：从 Tabular Q-Learning 过渡到 Deep Reinforcement Learning，理解为什么需要函数逼近、DQN 的输入输出、Replay Buffer、Target Network、Batch Training、TD Target、Episode/Transition、探索与学习之间的关系，并完成 CartPole-v1 上的完整 DQN 训练与可视化。

---

## 1. 为什么需要从 Q-table 走向 DQN

在 Task 3 中，Q-Learning 通过维护一张 Q-table：

$$
Q(s,a)
$$

来记录每一个状态-动作对的长期价值。

更新公式：

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_t+\gamma\max_aQ(s_{t+1},a)-Q(s_t,a_t)
\right]
$$

其中：

$$
r_t+\gamma\max_aQ(s_{t+1},a)
$$

是 TD Target。

Tabular Q-Learning 的问题在于：状态空间一旦变大、变成连续值、图像或高维传感器输入，就无法枚举所有状态。

例如：

- GridWorld：状态数量有限，可以直接查表；
- CartPole：状态是连续向量；
- Atari：状态可以是一张图像；
- UAV：状态可能包含位置、速度、电量、频谱、信道、环境观测等。

此时不可能继续维护：

```python
Q[state, action]
```

所以需要把 Q 函数改为神经网络：

$$
\boxed{
Q(s,a)
\rightarrow
Q_\theta(s,a)
}
$$

这里的 \(\theta\) 是神经网络参数。

需要特别注意：

> DQN 不是先得到一张“真实 Q-table”，再让神经网络去拟合它。

DQN 中已经没有显式的 Q-table，神经网络本身就是 Q 函数的近似表示。

---

# 2. 从 Tabular Q-Learning 到 Neural Q-Learning

Tabular Q-Learning 中，可以直接修改：

```python
Q[state, action]
```

但 DQN 中：

$$
Q(s,a)=Q_\theta(s,a)
$$

Q 值是神经网络前向传播得到的，因此不能直接修改某个 Q 值，只能通过 loss 和反向传播修改参数 \(\theta\)。

对于一条 transition：

$$
(s_t,a_t,r_t,s_{t+1})
$$

当前预测：

$$
\hat y_t=Q_\theta(s_t,a_t)
$$

TD Target：

$$
y_t=r_t+\gamma\max_{a'}Q(s_{t+1},a')
$$

于是可以构造监督学习形式的 loss：

$$
L(\theta)
=
\left[
Q_\theta(s_t,a_t)-y_t
\right]^2
$$

再通过：

```python
loss.backward()
optimizer.step()
```

更新神经网络参数。

因此，两种方法的核心对应关系为：

```text
Tabular Q-Learning                 DQN

Q[s, a]                            Qθ(s, a)
   ↓                                  ↓
TD Target                          TD Target
   ↓                                  ↓
TD Error                           Loss
   ↓                                  ↓
直接修改 Q[s,a]                    Backpropagation
                                      ↓
                                  更新参数 θ
```

最核心的变化只有：

$$
\boxed{
Q\text{-table}
\rightarrow
Q\text{-Network}
}
$$

Q-Learning 的 Bellman / TD 更新思想本身没有改变。

---

# 3. TD、SARSA、Q-Learning 与 DQN 的关系

一个重要的概念修正：

> Q-Learning 本身就是 TD Learning 的一种。

更准确的关系：

```text
TD Learning
│
├── TD(0) Prediction
│      └── 学习 V(s)
│
├── SARSA
│      └── On-policy TD Control
│
└── Q-Learning
       └── Off-policy TD Control
              │
              └── DQN
                   └── Deep Q-Learning
```

## 3.1 SARSA

SARSA 使用下一步实际执行的动作：

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)
\right]
$$

它关心：

> 按照当前策略，下一步实际上会做什么。

因此是 On-policy。

---

## 3.2 Q-Learning

Q-Learning 不管下一步实际执行什么，而使用：

$$
\max_{a'}Q(s_{t+1},a')
$$

更新：

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)
\right]
$$

因此它学习的是“下一步假设采取当前最优动作”的价值，是 Off-policy 方法。

---

## 3.3 DQN

DQN 继承的就是 Q-Learning：

$$
Q(s,a)
\rightarrow
Q_\theta(s,a)
$$

因此：

$$
\boxed{
DQN = Q\text{-Learning} + Neural Network + 稳定训练技巧
}
$$

---

# 4. Q Network 的输入与输出

这是 Task 4 中最容易混淆、也最关键的部分之一。

对于离散动作空间，经典 DQN 通常设计为：

$$
\boxed{
s
\rightarrow
[
Q(s,a_1),
Q(s,a_2),
\dots,
Q(s,a_n)
]
}
$$

也就是说：

- 输入：当前状态 \(s\)
- 输出：当前状态下所有离散动作对应的 Q 值

假设动作：

```text
0 = left
1 = right
2 = up
3 = down
```

网络输入状态 \(s\)，输出：

```text
state
  ↓
Q Network
  ↓
[Q(left), Q(right), Q(up), Q(down)]
```

例如：

```text
[1.2, 4.5, 2.1, 0.3]
```

表示：

$$
Q(s,left)=1.2
$$

$$
Q(s,right)=4.5
$$

$$
Q(s,up)=2.1
$$

$$
Q(s,down)=0.3
$$

如果采用 greedy：

$$
a=\arg\max_aQ(s,a)
$$

则选择 `right`。

---

## 4.1 CartPole 的例子

CartPole 的状态：

$$
s=
[
x,
\dot{x},
\theta,
\dot{\theta}
]
$$

即：

- 小车位置
- 小车速度
- 杆角度
- 杆角速度

因此：

$$
state\_dim=4
$$

动作有两个：

- 0：left
- 1：right

因此：

$$
action\_dim=2
$$

网络结构：

$$
\mathbb R^4
\rightarrow
\mathbb R^2
$$

也就是：

```text
[x, velocity, angle, angular_velocity]
                  ↓
              Q Network
                  ↓
          [Q(left), Q(right)]
```

---

## 4.2 DQN 不是 Policy Network

DQN：

```text
state
  ↓
Q Network
  ↓
Q values
```

Policy Network 更像：

```text
state
  ↓
Policy Network
  ↓
action probabilities
```

Dynamics Model 则更像：

```text
state + action
      ↓
Environment Model
      ↓
next_state
```

Task 4 中学习的是第一种。

---

# 5. 神经网络怎么“对齐”不同状态的 Q 值

Q-table 中状态的对齐依赖数组索引：

```python
Q[state]
```

神经网络中没有显式“第几行状态”的概念。

DQN 学习的是一个函数：

$$
Q_\theta(s,\cdot)
$$

给定任何状态 \(s\)，网络都会根据该状态的特征输出对应 Q 值。

因此对齐关系来自：

$$
\boxed{
s
\rightarrow
Q_\theta(s,\cdot)
}
$$

而不是：

```text
state ID → Q-table 某一行
```

例如三个完全不同状态：

```text
A
X
C
```

组成 batch 后：

```text
A → Qθ(A,·)
X → Qθ(X,·)
C → Qθ(C,·)
```

神经网络只是并行执行了多个函数计算。

---

# 6. 函数逼近为什么既有优势又有风险

Q-table 中：

```python
Q[A, right]
```

被更新时，其他状态的 Q 值基本不会受到影响。

DQN 中，所有状态共享同一组网络参数：

$$
\theta
$$

一次：

```python
optimizer.step()
```

可能同时改变很多状态的 Q 值。

这带来两个结果：

## 优点：泛化

如果两个状态非常相似：

```text
s1 = [10.0, 20.0]
s2 = [10.1, 20.0]
```

Q-table 会认为它们是完全独立的状态。

神经网络则可能利用相似特征进行泛化。

---

## 缺点：相互干扰

为了让：

$$
Q(A,right)
$$

变得更准确，更新共享参数时，可能导致：

$$
Q(B,left)
$$

等其他状态的预测发生变化。

因此：

$$
\boxed{
共享参数
=
泛化能力
+
潜在训练不稳定
}
$$

---

# 7. Transition 是什么

一条 transition 是一次完整的单步交互：

$$
\boxed{
(s_t,a_t,r_t,s_{t+1},done)
}
$$

含义：

```text
当前状态 s_t
    ↓
执行动作 a_t
    ↓
环境返回奖励 r_t
    ↓
进入下一状态 s_{t+1}
    ↓
是否终止 done
```

Transition 可以理解为“一步经验”。

---

# 8. Episode 是什么

Episode 是从一次环境 reset 开始，到这次任务结束为止的一整次任务尝试。

例如：

```text
reset
↓
transition 1
↓
transition 2
↓
transition 3
↓
...
↓
Goal / Failure / Time Limit
↓
Episode End
```

因此：

$$
\boxed{
1\ Episode
\supset
很多\ Transitions
}
$$

例如：

```text
Episode 1 → 5 transitions
Episode 2 → 10 transitions
Episode 3 → 7 transitions
```

Episode 长度完全可以不同。

标准 DQN 存的是单步 transition，因此不需要把不同 episode padding 到相同长度。

---

# 9. Episode 与 Epoch 不同

监督学习中的 Epoch：

> 整个固定 Dataset 被完整遍历一次。

DQN 中的数据集不是固定的。

Replay Buffer 会不断：

- 加入新的 transition
- 删除过旧 transition
- 随机采样历史经验

因此 RL 中更常使用：

### Environment Step

执行一次动作：

$$
s_t\xrightarrow{a_t}s_{t+1}
$$

产生一条 transition。

### Gradient Step

从 Replay Buffer 采样一个 batch，执行一次：

```python
loss.backward()
optimizer.step()
```

### Episode

从 reset 到终止的一整次任务。

因此：

$$
\boxed{
Episode \neq Epoch
}
$$

---

# 10. terminated 与 truncated

现代 Gymnasium 会返回：

```python
next_state, reward, terminated, truncated, info
```

## terminated

环境真正进入终止状态。

例如：

- 到达 Goal
- 掉入陷阱
- CartPole 杆倒下
- 游戏角色死亡

---

## truncated

环境本身未必进入真正的 terminal state，而是由于外部限制结束。

例如：

- 达到最大步数
- 达到最大时间
- CartPole 成功坚持到时间上限

Episode 是否结束：

```python
terminated or truncated
```

但在严格的 TD bootstrap 中，应区分真正的 terminal 和时间截断。

---

# 11. DQN 是 Model-Free，但为什么能得到 Reward

这是本任务中一个重要问题。

Model-Free 并不是：

> Agent 连 reward 和 next_state 都不知道。

Model-Free 的真正含义是：

> Agent 不需要事先知道环境模型。

也就是不需要知道：

$$
P(s'|s,a)
$$

以及完整奖励模型：

$$
R(s,a)
$$

但是当 Agent 真正执行动作后，环境会返回：

$$
s_{t+1}
$$

以及：

$$
r_t
$$

即：

```text
Agent 不知道：
“执行 right 会发生什么？”

Agent 执行 right

Environment 返回：
next_state = B
reward = -1
```

Agent 得到的是一次 sample：

$$
(A,right,-1,B)
$$

但它仍然不知道完整的：

$$
P(s'|A,right)
$$

因此：

$$
\boxed{
观察一次 transition
\neq
知道环境模型
}
$$

---

## 11.1 Model-Free 与 Model-Based

Model-Free：

```text
执行真实动作
↓
观察 transition
↓
直接学习 Q / Policy
```

Model-Based：

```text
知道或学习环境模型
P(s'|s,a)
R(s,a)
↓
可以预测“如果执行某动作会怎样”
↓
利用模型规划
```

DQN 不显式学习：

$$
P(s'|s,a)
$$

因此属于 Model-Free RL。

---

# 12. 为什么不能简单“遍历一次环境”

对于很小的确定性 GridWorld：

> 理论上确实可以。

例如：

$$
15\ states\times4\ actions=60
$$

如果所有 state-action pair 都可枚举，并且环境确定，可以得到完整转移模型，再使用 Dynamic Programming。

但真实 RL 问题通常：

- 状态空间巨大；
- 状态连续；
- 状态由图像表示；
- 转移具有随机性；
- 不同状态无法全部访问。

例如 CartPole：

$$
s=[x,\dot x,\theta,\dot\theta]
$$

都是连续变量，理论状态数近似无限。

因此无法：

```text
遍历所有状态
```

这也是函数逼近存在的根本意义。

---

# 13. TD Target 到底是什么

DQN 最终真正希望学习的是：

$$
Q^*(s,a)
$$

也就是状态-动作对的真实长期回报期望。

定义：

$$
Q^\pi(s,a)
=
\mathbb E[G_t|s_t=s,a_t=a]
$$

其中：

$$
G_t
=
r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
$$

问题是：

> 在当前时刻，我们通常不知道未来所有 reward。

所以无法直接获得真实 \(Q^*(s,a)\) 作为监督标签。

于是 TD 方法使用：

$$
y_t
=
r_t+\gamma\max_{a'}Q(s_{t+1},a')
$$

作为临时训练目标。

因此：

$$
\boxed{
TD\ Target
=
真实的一步 Reward
+
估计的未来价值
}
$$

其中：

- \(r_t\)：环境真实返回；
- \(\max Q(s_{t+1},a)\)：模型估计。

---

# 14. TD Target 不是真实 Q 值

例如：

```text
A --right--> B
reward = 1
```

当前：

$$
Q(A,right)=3
$$

网络对 B 的预测：

$$
Q(B,\cdot)=[2,5,3,1]
$$

则：

$$
\max_aQ(B,a)=5
$$

若：

$$
\gamma=0.9
$$

则：

$$
target
=
1+0.9\times5
=
5.5
$$

于是：

```text
prediction = 3
target     = 5.5
```

网络会让：

$$
Q(A,right)
$$

往 5.5 的方向更新。

但需要强调：

> 5.5 不是真实答案，只是当前基于 reward 和下一状态 Q 估计构造出来的学习目标。

随着下一状态 Q 估计不断改善，前面状态的 Q 值也会逐渐改善。

---

# 15. Bootstrap

Bootstrap 的核心：

> 用当前已经估计出来的价值，帮助更新另一个价值。

DQN：

$$
Q(s,a)
\leftarrow
r+\gamma\max Q(s',a')
$$

其中：

$$
Q(s',a')
$$

本身也是估计出来的。

所以：

```text
估计未来状态
↓
构造当前状态 Target
↓
更新当前状态
```

这就是 Bootstrap。

优点：

- 不需要等 episode 完成；
- 可以每一步更新。

缺点：

- 估计误差可能传播；
- target 自身不是完全真实的；
- 神经网络训练容易不稳定。

---

# 16. Monte Carlo 与 TD Target 的区别

假设：

```text
A → B → C → Goal
```

奖励：

```text
A → B : 1
B → C : 2
C → G : 10
```

若：

$$
\gamma=0.9
$$

Monte Carlo 可以等整个 Episode 结束后直接计算：

$$
G_A
=
1+0.9\times2+0.9^2\times10
$$

这相当于使用实际完整 return。

而 TD/Q-Learning 在刚刚经历：

```text
A → B
```

时，就可以立即更新：

$$
target
=
1+\gamma\max Q(B,a)
$$

因此：

```text
MC：
等待整局完成
↓
使用完整 Return

TD：
只走一步
↓
使用 Reward + Bootstrap
↓
立刻更新
```

---

# 17. Batch 在 DQN 中是什么意思

DQN 的 batch 和普通深度学习中的 mini-batch 在张量训练方式上基本一致，但数据来源不同。

监督学习：

```text
固定 Dataset
↓
随机 Batch
↓
Neural Network
```

DQN：

```text
Environment
↓
Transitions
↓
Replay Buffer
↓
随机 Batch
↓
Q Network
```

假设：

$$
batch\_size=32
$$

$$
state\_dim=4
$$

$$
action\_dim=2
$$

则：

```text
states
[32, 4]

actions
[32]

rewards
[32]

next_states
[32, 4]

terminateds
[32]
```

网络：

```text
states [32,4]
      ↓
Q Network
      ↓
q_values [32,2]
```

---

# 18. 为什么 actions 是 [B] 而不是 [B, action_dim]

`actions` 保存的是：

> 每条 transition 当时实际执行了哪个动作。

例如：

```python
actions = [1, 0, 1, 1, 0, ...]
```

因此：

$$
actions.shape=[B]
$$

而：

$$
Q_\theta(states)
$$

输出的是每个状态的所有动作 Q 值：

$$
[B,action\_dim]
$$

然后根据实际执行的动作，通过：

```python
gather()
```

得到：

$$
Q(s_i,a_i)
$$

shape：

$$
[B]
$$

因此：

```text
Q values:
[B, A]

actions:
[B]

gather

selected Q:
[B]
```

---

# 19. Batch 不要求状态相同，也不要求 Episode 相同

Replay Buffer 可以随机抽出：

```text
Episode 3 的 step 4
Episode 17 的 step 2
Episode 8 的 step 10
...
```

这些 transition：

- 起点可以不同；
- 终点可以不同；
- episode 可以不同；
- 时间位置可以不同；
- 状态也可以重复。

唯一要求：

> 每一行中的 \(s,a,r,s',done\) 必须属于同一条真实 transition。

即：

$$
(s_i,a_i,r_i,s_i',d_i)
$$

内部必须正确对应。

不同 batch 行之间不需要任何时间对齐。

---

# 20. Replay Buffer

连续 RL 数据高度相关。

例如：

```text
s1 → s2
s2 → s3
s3 → s4
```

如果网络每次只使用最新 transition：

```text
transition 1 → update
transition 2 → update
transition 3 → update
```

训练数据相关性非常强。

因此 DQN 引入 Replay Buffer：

```text
Environment
↓
Transition
↓
Replay Buffer
↓
Random Sample Batch
↓
Network Update
```

主要作用：

## 20.1 打破时间相关性

随机抽历史经验，让 batch 中的数据来自不同时间和 episode。

## 20.2 重复利用经验

历史 transition 可以被多次采样使用，而不是一次更新后就丢弃。

---

# 21. Replay Buffer 与状态覆盖问题

Replay Buffer 的 uniform random sampling：

> 对 Buffer 中已有 transition 等概率抽样。

它不代表：

> 对整个状态空间均匀采样。

例如 Buffer 中：

```text
A 状态经验：500
B 状态经验：300
C 状态经验：150
D 状态经验：50
```

那么 uniform replay 仍然更容易抽到 A。

因此：

$$
\boxed{
Uniform\ Replay
\neq
Uniform\ State\ Visitation
}
$$

---

# 22. 谁负责探索状态空间

状态和 state-action coverage 主要由行为策略决定。

DQN 使用：

$$
\epsilon\text{-greedy}
$$

$$
a=
\begin{cases}
random,&p<\epsilon\\
\arg\max_aQ(s,a),&otherwise
\end{cases}
$$

其中：

```text
ε-greedy
负责：
“去哪里收集数据？”
```

Replay Buffer：

```text
负责：
“已有数据中拿哪些来训练？”
```

因此：

```text
Exploration
↓
决定收集什么 transition
↓
Replay Buffer
↓
决定利用哪些已有 transition
```

---

# 23. 为什么更应该关注 State-Action Coverage

即使一个状态 A 被访问很多次，也不代表所有动作都学好了。

例如：

```text
A → right : 999 次
A → up    : 1 次
A → left  : 0 次
A → down  : 0 次
```

虽然状态 A 被访问 1000 次，但：

$$
Q(A,left)
$$

几乎没有学习数据。

因此实际更关心：

$$
\boxed{
state-action\ coverage
}
$$

而不只是 state coverage。

---

# 24. Target Network

如果 prediction 和 target 都使用同一个快速更新的网络：

$$
Q_\theta(s,a)
$$

以及：

$$
r+\gamma\max Q_\theta(s',a)
$$

每次：

```python
optimizer.step()
```

都会改变 \(\theta\)。

于是：

- prediction 在变化；
- target 也跟着快速变化。

训练目标不断移动，容易导致不稳定。

因此 DQN 引入两个网络：

$$
\boxed{
Online\ Network: Q_\theta
}
$$

$$
\boxed{
Target\ Network: Q_{\theta^-}
}
$$

---

# 25. Online Network 与 Target Network 的分工

## Online Network

负责：

- 当前 Q 值预测；
- \(\epsilon\)-greedy 动作选择；
- 参与反向传播；
- 每个 gradient step 更新。

## Target Network

负责：

- 计算下一状态 Q；
- 构造 TD Target；
- 不参与正常梯度更新；
- 每隔一定频率同步一次。

因此：

$$
prediction
=
Q_\theta(s,a)
$$

而：

$$
target
=
r+\gamma\max_aQ_{\theta^-}(s',a)
$$

---

# 26. Target Network 里的 Q 也是估计值

Target Network 并不知道真实 Q 值。

它仍然是一个神经网络：

$$
Q_{\theta^-}(s,a)
$$

其输出仍是估计值。

Target Network 的价值在于：

> 它是一个变化得更慢的估计器，而不是一个真实答案提供者。

---

# 27. Target Network 如何更新

典型 Hard Update：

$$
\boxed{
\theta^-\leftarrow\theta
}
$$

代码：

```python
target_network.load_state_dict(
    q_network.state_dict()
)
```

需要注意：

> 更新的不是某一个 Q 值，而是整个 Online Network 的参数。

同步瞬间：

$$
\theta^-=\theta
$$

因此对同一个输入：

$$
Q_{\theta^-}(s,a)
=
Q_\theta(s,a)
$$

但下一次 Online Network 更新后：

$$
\theta\rightarrow\theta'
$$

Target Network 仍然保持旧参数。

---

# 28. 为什么 Online 和 Target 参数相同也能继续学习

非常关键：

$$
\theta=\theta^-
$$

并不意味着：

$$
Q_\theta(s,a)
=
r+\gamma Q_{\theta^-}(s',a')
$$

例如同步后：

$$
Q(A,right)=3
$$

下一状态：

$$
Q(B,\cdot)=[2,5,3,1]
$$

Reward：

$$
r=1
$$

则：

$$
target=1+0.9\times5=5.5
$$

所以：

```text
prediction = 3
target     = 5.5
```

loss 仍然不为 0。

因此：

$$
\boxed{
Online=Target
\neq
Bellman\ Error=0
}
$$

Target Network 同步不代表网络已经收敛。

---

# 29. 真正的“接近收敛”是什么

真正需要接近的是：

$$
Q(s,a)
\approx
r+\gamma\max_aQ(s',a)
$$

更严格地说，最优 Q 函数满足 Bellman Optimality Equation：

$$
Q^*(s,a)
=
\mathbb E
\left[
r+\gamma\max_{a'}Q^*(s',a')
\right]
$$

而不是简单满足：

$$
Online=Target
$$

同时，在深度 RL 中不能仅看 loss 判断收敛，还应该观察：

- Average Return
- Success Rate
- Episode Length
- Evaluation Performance

---

# 30. `torch.no_grad()` 与 Target Network 的区别

`torch.no_grad()`：

> 只保证当前这一次反向传播中，target 分支不参与梯度计算。

Target Network：

> 保证多个训练 step 之间，target 不会变化得太快。

因此二者作用不同：

```text
no_grad
↓
防止 target 分支参与当前 BP

Target Network
↓
降低跨多个训练 step 的 moving target
```

---

# 31. Terminal State 的 TD Target

一般 TD Target：

$$
y
=
r+\gamma(1-done)\max_aQ(s',a)
$$

如果：

$$
done=0
$$

则：

$$
y=r+\gamma\max Q(s',a)
$$

如果真正 terminal：

$$
done=1
$$

则：

$$
y=r
$$

因为 terminal state 后面已经没有未来 reward。

---

# 32. DQN 的完整 Batch Update

假设 Replay Buffer 随机抽出：

$$
B
$$

条 transition。

### Step 1：当前状态

$$
states:[B,state\_dim]
$$

经过 Online Network：

$$
Q_\theta(states):[B,action\_dim]
$$

根据实际 actions：

$$
Q_\theta(s_i,a_i):[B]
$$

---

### Step 2：下一状态

$$
next\_states:[B,state\_dim]
$$

经过 Target Network：

$$
Q_{\theta^-}(next\_states):[B,action\_dim]
$$

每一行取：

$$
\max_aQ_{\theta^-}(s_i',a)
$$

得到：

$$
[B]
$$

---

### Step 3：构造 Target

$$
y_i
=
r_i+\gamma(1-d_i)\max_aQ_{\theta^-}(s_i',a)
$$

得到：

$$
targets:[B]
$$

---

### Step 4：计算 Loss

$$
L
=
\frac1B
\sum_{i=1}^{B}
\left[
Q_\theta(s_i,a_i)-y_i
\right]^2
$$

然后：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

一次 batch：

$$
\boxed{
B\ 条\ transition
\rightarrow
B\ 个\ prediction
\rightarrow
B\ 个\ target
\rightarrow
1\ 个\ batch\ loss
\rightarrow
1\ 次\ gradient\ update
}
$$

---

# 33. DQN 的完整训练流程

整体结构：

```text
Current State
     ↓
Online Q Network
     ↓
ε-greedy
     ↓
Action
     ↓
Environment
     ↓
Reward + Next State
     ↓
Transition
(s, a, r, s', done)
     ↓
Replay Buffer
     ↓
Random Batch
     ↓
┌──────────────────────────┐
│ Online Network           │
│ Qθ(s,a)                  │
│ prediction               │
└───────────┬──────────────┘
            │
            │ loss
            │
┌───────────┴──────────────┐
│ Target Network           │
│ r + γ max Qθ-(s',a)      │
│ TD target                │
└──────────────────────────┘
            ↓
       Backpropagation
            ↓
   Update Online Network

每隔 N 次：
Online θ → Target θ-
```

---

# 34. DQN 是边探索边学习

DQN 不是：

```text
先把环境完全探索完
↓
再统一训练
```

更典型的是：

```text
与环境交互
↓
产生 transition
↓
存入 Replay Buffer
↓
随机采 batch
↓
更新网络
↓
继续交互
↓
继续学习
```

因此：

$$
\boxed{
Collect\ Data
\leftrightarrow
Learn
}
$$

交替进行。

---

# 35. Learning Starts / Warm-up

训练开始时 Replay Buffer 太空。

如果只有少量高度相关 transition 就立刻训练，效果不好。

所以常用：

```python
LEARNING_STARTS = 1000
```

即：

```text
前 1000 条 transition：
主要收集经验

Buffer 足够大之后：
开始边收集边训练
```

这叫 warm-up。

注意：

> Warm-up 是“先积累一部分经验”，不是“先探索完整环境”。

---

# 36. Epsilon Decay

训练开始时网络基本随机，因此需要更多探索：

$$
\epsilon\approx1
$$

随着训练进行，逐渐降低：

$$
\epsilon\rightarrow0.05
$$

典型过程：

```text
训练初期：
ε ≈ 1.0
→ 主要探索

训练中期：
ε ≈ 0.5
→ 探索 + 利用

训练后期：
ε ≈ 0.05
→ 主要利用
```

需要注意：

> 后期 Episode 变长，不代表探索更多。

实际上通常：

$$
\epsilon
$$

越来越小。

Episode 变长是因为策略学得更好，Agent 能完成更多有效 step。

---

# 37. 为什么后期每个 Episode 运行更慢

在 CartPole 中：

- 前期策略差，可能 20 步就结束；
- 后期策略好，可能坚持 500 步。

当前实现基本：

```text
1 environment step
≈
1 transition
≈
1 network update
```

因此：

```text
早期 Episode：
20 steps
≈ 20 次 update

后期 Episode：
500 steps
≈ 500 次 update
```

所以一个后期 Episode 的计算量可能远大于前期 Episode。

这不是程序“越来越卡”，而是：

$$
\boxed{
Episode\ Length\ 增长
\rightarrow
Transition\ 数量增加
\rightarrow
Gradient\ Update\ 次数增加
}
$$

---

# 38. CartPole 中 Episode Reward 的含义

CartPole 中每存活一个 step 通常获得正奖励。

因此：

$$
Episode\ Reward
$$

基本和：

$$
Episode\ Length
$$

高度一致。

策略越好：

```text
杆保持时间越长
↓
Episode 越长
↓
Reward 越高
```

当达到环境时间上限时：

```text
truncated=True
```

Episode 结束。

---

# 39. 如何判断 DQN 是否学得好

不能只看 TD Loss。

因为：

$$
target
$$

本身也在不断变化。

更重要的是观察：

## Episode Reward / Average Return

最核心指标。

## Episode Length

对 CartPole 很直观。

## Evaluation Reward

测试时：

$$
\epsilon=0
$$

完全 greedy，观察训练好的策略真实表现。

## TD Loss

用于诊断训练，但不能单独代表策略性能。

---

# 40. 为什么 DQN Loss 不一定单调下降

监督学习：

```text
Prediction
vs
固定 Label
```

DQN：

```text
Prediction
vs
动态 TD Target
```

Target 会受到：

- Target Network 参数；
- Replay Buffer 数据；
- 网络估计；
- 行为策略变化；
- 状态访问分布变化；

等因素影响。

因此 TD Loss：

- 可能震荡；
- 可能阶段性升高；
- 不一定趋近 0。

所以：

$$
\boxed{
Reward\ Curve
>
Loss\ Curve
}
$$

在评价策略时更重要。

---

# 41. Task 4 中的 CartPole 实验结构

推荐 Notebook：

```text
Task 04 — DQN

1. Import
2. Seed + Device
3. Environment
4. Epsilon Decay
5. Hyperparameters
6. Create DQNAgent
7. Training
8. Moving Average
9. Reward Curve
10. TD Loss Curve
11. Epsilon Curve
12. Training Summary
13. Evaluation
14. Evaluation Plot
15. Inspect Online / Target Q-values
```

工程代码继续放在：

```text
src/task04_dqn/
├── q_network.py
├── replay_buffer.py
└── dqn_agent.py
```

Notebook 只负责：

- 实验；
- 调参；
- 训练；
- 可视化；
- 结果分析。

---

# 42. Task 4 核心代码对应公式

## 动作选择

```python
action = agent.select_action(state)
```

对应：

$$
\epsilon\text{-greedy}
$$

---

## 环境交互

```python
next_state, reward, terminated, truncated, info = env.step(action)
```

产生：

$$
(s,a,r,s')
$$

---

## 保存 Transition

```python
agent.store_transition(...)
```

进入 Replay Buffer。

---

## Online Prediction

```python
q_values = q_network(states)

current_q = q_values.gather(
    1,
    actions.unsqueeze(1)
).squeeze(1)
```

对应：

$$
Q_\theta(s_i,a_i)
$$

---

## Target Network

```python
with torch.no_grad():
    next_q_values = target_network(next_states)
    max_next_q = next_q_values.max(dim=1).values
```

对应：

$$
\max_aQ_{\theta^-}(s_i',a)
$$

---

## TD Target

```python
target_q = (
    rewards
    + gamma
    * (1 - terminateds)
    * max_next_q
)
```

对应：

$$
y_i
=
r_i+\gamma(1-d_i)\max_aQ_{\theta^-}(s_i',a)
$$

---

## Loss

```python
loss = F.mse_loss(current_q, target_q)
```

对应：

$$
L
=
\frac1B
\sum_i
\left(
Q_\theta(s_i,a_i)-y_i
\right)^2
$$

---

## Backpropagation

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

对应：

$$
\theta
\leftarrow
\theta-\eta\nabla_\theta L
$$

---

## Target Update

```python
target_network.load_state_dict(
    q_network.state_dict()
)
```

对应：

$$
\theta^-\leftarrow\theta
$$

---

# 43. Task 2 → Task 3 → Task 4 的知识链

## Task 2：Dynamic Programming

假设知道环境模型：

$$
P(s'|s,a)
$$

$$
R(s,a)
$$

通过 Bellman Equation 做：

- Policy Evaluation
- Policy Iteration
- Value Iteration

---

## Task 3：Model-Free Tabular RL

不知道完整环境模型。

通过真实 interaction 获得：

$$
(s,a,r,s')
$$

学习：

- Monte Carlo
- TD
- SARSA
- Q-Learning

其中 Q-Learning：

$$
Q\text{-table}
$$

仍要求离散且有限状态空间。

---

## Task 4：Deep Reinforcement Learning

继续保持：

$$
Model-Free
$$

但进一步：

$$
Q\text{-table}
\rightarrow
Q_\theta(s,a)
$$

从而处理连续、高维、大规模状态空间。

因此整体演化路线：

```text
Known Model
   ↓
Dynamic Programming
   ↓
Unknown Model
   ↓
MC / TD
   ↓
SARSA / Q-Learning
   ↓
Large / Continuous State Space
   ↓
Function Approximation
   ↓
DQN
```

---

# 44. 本任务中几个最重要的易错点

## 误区 1：DQN 是拿神经网络拟合一张已有 Q-table

错误。

正确理解：

$$
Q_\theta(s,a)
$$

本身就是 Q 函数。

---

## 误区 2：DQN 的真实 target 是已知的

错误。

TD Target 只是：

$$
r+\gamma\max Q(s',a)
$$

构造出来的临时监督信号。

真正想逼近的是：

$$
Q^*(s,a)
$$

---

## 误区 3：Target Network 给的是“真实答案”

错误。

Target Network 里的 Q 同样是估计值。

它只是一个更新得更慢的 Q 网络。

---

## 误区 4：Online 和 Target 同步意味着网络收敛

错误。

同步只意味着：

$$
\theta^-=\theta
$$

真正重要的是：

$$
Q(s,a)
$$

是否已经接近正确长期回报。

---

## 误区 5：Batch 必须来自相同状态或相同 Episode

错误。

Batch 中不同 transition 完全可以：

- 来自不同 episode；
- 来自不同时间；
- 来自不同状态。

---

## 误区 6：actions shape 应该是 [B, action_dim]

通常错误。

经典 DQN 中动作直接使用整数索引：

$$
actions:[B]
$$

Q Network 输出：

$$
[B,A]
$$

通过 gather 取得：

$$
[B]
$$

---

## 误区 7：Replay Buffer 能保证均匀访问所有状态

错误。

Replay Buffer 只负责从已有经验中采样。

探索范围主要取决于行为策略。

---

## 误区 8：Episode 相当于 Epoch

错误。

Episode 是一次完整任务尝试。

Epoch 是监督学习中完整遍历固定 Dataset 一次。

---

## 误区 9：Model-Free 不能知道 reward

错误。

Model-Free 不需要提前知道完整奖励模型，但执行真实动作后可以观察：

$$
r_t
$$

---

## 误区 10：DQN 先探索完，再训练

通常错误。

更典型的是：

$$
\boxed{
Interaction
\leftrightarrow
Learning
}
$$

只是在训练最开始有短暂 warm-up。

---

# 45. Task 4 最终总结

DQN 的核心可以浓缩为：

$$
\boxed{
Q\text{-Learning}
+
Function\ Approximation
+
Replay\ Buffer
+
Target\ Network
+
\epsilon\text{-greedy}
}
$$

其中：

### Q-Learning

提供 TD / Bellman 更新逻辑。

### Q Network

解决大规模、连续状态空间下 Q-table 无法枚举的问题。

### Replay Buffer

打破连续 transition 相关性，并重复利用历史经验。

### Target Network

减缓 TD Target 变化速度，提高训练稳定性。

### ε-greedy

在 exploration 和 exploitation 之间进行权衡。

完整逻辑：

```text
state
↓
Online Q Network
↓
ε-greedy
↓
action
↓
Environment
↓
reward + next_state
↓
transition
↓
Replay Buffer
↓
random batch
↓
Online Network → prediction
Target Network → TD target
↓
loss
↓
backprop
↓
update Online
↓
periodically sync Target
```

最终目标始终没有改变：

$$
\boxed{
学习尽可能准确的长期状态-动作价值
Q^*(s,a)
}
$$

Task 4 的真正意义，是完成了强化学习中最重要的一次表示方式转换：

$$
\boxed{
表格型价值函数
\rightarrow
神经网络价值函数
}
$$

从这里开始，后续 Deep RL 算法都建立在“用神经网络表示 RL 中的价值函数、策略或模型”这一基本思想之上。


---

## Task 04 实验记录

固定 CartPole、网络宽度和 seed，对比 replay buffer + target network 的完整 DQN 与移除一个稳定化组件的版本，记录训练/评估回报和 loss。

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
