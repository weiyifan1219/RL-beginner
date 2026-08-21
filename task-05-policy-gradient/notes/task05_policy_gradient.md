# Task 05：Policy Gradient

> 本任务目标：从 Value-based RL 过渡到 Policy-based RL，理解策略梯度的核心思想、REINFORCE、Baseline 与 Advantage，并通过 CartPole 实验验证 Baseline 对训练稳定性和样本效率的改善。

---

## 1. 从 DQN 到 Policy Gradient

在 Task 4 中，我们学习的是 DQN。DQN 本质上仍然属于 Value-based RL：

$$
Q_\theta(s,a)
$$

网络输入状态 \(s\)，输出每个动作的价值：

$$
[Q(s,a_1),Q(s,a_2),\dots]
$$

再通过：

$$
a=\arg\max_a Q(s,a)
$$

间接得到策略。

因此：

```text
DQN
State
  ↓
Q Network
  ↓
Q(s,a)
  ↓
argmax
  ↓
Action
```

Policy Gradient 则直接学习：

$$
\pi_\theta(a|s)
$$

即：

> 在状态 \(s\) 下，每个动作应该以多大的概率被选择。

例如：

```text
left   : 0.10
right  : 0.15
up     : 0.70
down   : 0.05
```

所以 Policy-based RL 的流程是：

```text
State
  ↓
Policy Network
  ↓
πθ(a|s)
  ↓
Categorical Sampling
  ↓
Action
```

### DQN 与 Policy Gradient 的核心区别

| 方法 | 学什么 | 如何得到动作 |
|---|---|---|
| DQN | \(Q_\theta(s,a)\) | \(\arg\max Q(s,a)\) |
| Policy Gradient | \(\pi_\theta(a|s)\) | 按概率分布采样 |

DQN 通常通过 \(\epsilon\)-greedy 加入探索，而随机 Policy 本身就具有探索性。

---

## 2. Policy Network

离散动作空间中，Policy Network 可以写成：

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        logits = self.net(state)
        probs = torch.softmax(logits, dim=-1)
        return probs
```

网络先输出 logits：

$$
z_\theta(s)
$$

再通过 Softmax：

$$
\pi_\theta(a_i|s)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

得到动作概率。

### 为什么不用 argmax？

训练阶段采用：

```python
dist = Categorical(probs=probs)
action = dist.sample()
```

而不是：

```python
action = probs.argmax()
```

因为 Policy Gradient 需要从当前策略分布中采样轨迹。

---

## 3. 为什么要保存 `log_prob`

采样动作之后：

```python
log_prob = dist.log_prob(action)
```

得到：

$$
\log \pi_\theta(a_t|s_t)
$$

这是后面策略梯度更新的核心。

注意：

```python
log_prob = dist.log_prob(action)
```

不能写成：

```python
log_prob = dist.log_prob(action).item()
```

因为 `.item()` 会切断 PyTorch 计算图，导致无法通过：

```python
loss.backward()
```

更新 Policy Network。

---

# 4. Policy Gradient 的目标函数

策略优化的目标是最大化期望累积奖励：

$$
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
[R(\tau)]
$$

其中 trajectory：

$$
\tau=(s_0,a_0,s_1,a_1,\dots)
$$

展开期望：

$$
J(\theta)
=
\sum_\tau
P_\theta(\tau)R(\tau)
$$

对参数求梯度：

$$
\nabla_\theta J(\theta)
=
\sum_\tau
R(\tau)\nabla_\theta P_\theta(\tau)
$$

使用 log-derivative trick：

$$
\nabla_\theta P_\theta(\tau)
=
P_\theta(\tau)
\nabla_\theta\log P_\theta(\tau)
$$

得到：

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
R(\tau)
\nabla_\theta\log P_\theta(\tau)
\right]
$$

trajectory 概率可以写成：

$$
P_\theta(\tau)
=
p(s_0)
\prod_t
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t)
$$

其中只有 Policy：

$$
\pi_\theta(a_t|s_t)
$$

依赖参数 \(\theta\)。

最终得到经典 Policy Gradient：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\sum_t
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
}
$$

---

# 5. Return \(G_t\)

对于时间步 \(t\)：

$$
\boxed{
G_t
=
r_t+\gamma r_{t+1}
+\gamma^2r_{t+2}+\cdots
}
$$

也可以递推：

$$
\boxed{
G_t=r_t+\gamma G_{t+1}
}
$$

因此代码通常倒序计算：

```python
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)

    returns.reverse()

    return torch.tensor(
        returns,
        dtype=torch.float32,
    )
```

注意：

> 环境中的 trajectory 仍然是正向执行的，只是在 Episode 结束后，为了高效计算 Return，从后向前递推。

---

# 6. REINFORCE

REINFORCE 是最经典的 Monte Carlo Policy Gradient。

策略梯度：

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
G_t\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
$$

PyTorch 默认使用 Gradient Descent，因此定义：

$$
\boxed{
L_{\text{policy}}
=
-\sum_t
G_t
\log\pi_\theta(a_t|s_t)
}
$$

核心代码：

```python
loss = -(log_probs * returns).sum()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 为什么有负号？

Policy Gradient 要最大化：

$$
J(\theta)
$$

理论上应该做 Gradient Ascent：

$$
\theta
\leftarrow
\theta+\alpha\nabla_\theta J(\theta)
$$

但 PyTorch optimizer 默认做 Gradient Descent：

$$
\theta
\leftarrow
\theta-\alpha\nabla_\theta L(\theta)
$$

因此：

$$
L=-J
$$

---

# 7. REINFORCE 的完整训练过程

```text
Episode Start
      │
      ↓
    state
      │
      ↓
 Policy Network
      │
      ↓
 πθ(a|s)
      │
      ↓
 sample action
      │
      ├──── 保存 log_prob
      ↓
 env.step(action)
      │
      ├──── 保存 reward
      ↓
 next_state
      │
      ↓
   循环
      │
      ↓
Episode End
      │
      ↓
计算 G0,G1,...,GT
      │
      ↓
L = -Σ Gt log πθ(at|st)
      │
      ↓
backward()
      │
      ↓
optimizer.step()
```

原始 REINFORCE 的特点：

- 必须等完整 Episode 结束；
- 使用真实 Monte Carlo Return；
- 不使用 Bootstrapping；
- 梯度估计方差较高。

因此：

$$
\boxed{
\text{REINFORCE}
=
\text{Monte Carlo}
+
\text{Policy Gradient}
}
$$

---

# 8. 为什么 REINFORCE 方差大？

不同 trajectory 由随机 Policy 产生：

```text
Trajectory 1 → Return = 200
Trajectory 2 → Return = 80
Trajectory 3 → Return = 150
```

于是：

$$
G_t
$$

本身存在较大随机波动。

策略梯度：

$$
G_t\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

也会随 trajectory 大幅变化。

所以原始 REINFORCE 的典型问题是：

$$
\boxed{\text{High Variance}}
$$

---

# 9. Baseline

为了降低梯度估计方差，可以从 Return 中减掉一个 Baseline：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E
\left[
(G_t-b(s_t))
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
}
$$

只要 Baseline 不依赖 action，就不会改变策略梯度的期望。

最经典的 Baseline：

$$
\boxed{
b(s)=V(s)
}
$$

所以：

$$
G_t-V(s_t)
$$

表示：

> 这一次实际表现，相比当前策略在该状态下的正常水平，好了多少。

---

# 10. Advantage

理论定义：

$$
\boxed{
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
}
$$

含义：

- \(A(s,a)>0\)：动作比当前策略平均水平更好；
- \(A(s,a)<0\)：动作比平均水平更差；
- \(A(s,a)\approx0\)：动作大致符合当前策略正常表现。

在 REINFORCE + Baseline 中，我们没有单独训练 Q Network。

而是使用完整 trajectory 得到的：

$$
G_t
$$

作为：

$$
Q^\pi(s_t,a_t)
$$

的一次 Monte Carlo 样本估计。

所以实际代码中的 Advantage 是：

$$
\boxed{
\hat A_t
=
G_t-V_\phi(s_t)
}
$$

---

# 11. Value Network

Value Network 学习：

$$
V^\pi(s)
=
\mathbb E_\pi[G_t|S_t=s]
$$

即：

> 在状态 \(s\) 下，继续按照当前 Policy 行动，平均能获得多少 Return。

网络：

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)
```

Value Network 通过 Monte Carlo Return 作为监督目标：

$$
\boxed{
L_{\text{value}}
=
\frac1T
\sum_t
\left(
V_\phi(s_t)-G_t
\right)^2
}
$$

---

# 12. Policy Loss + Value Loss

REINFORCE + Baseline：

$$
\boxed{
\hat A_t
=
G_t-V_\phi(s_t)
}
$$

Policy Loss：

$$
\boxed{
L_{\text{policy}}
=
-\frac1T
\sum_t
\hat A_t
\log\pi_\theta(a_t|s_t)
}
$$

Value Loss：

$$
\boxed{
L_{\text{value}}
=
\frac1T
\sum_t
\left(
V_\phi(s_t)-G_t
\right)^2
}
$$

关键代码：

```python
values = value_network(states)

advantages = (
    returns
    - values.detach()
)

policy_loss = -(
    log_probs
    * advantages
).mean()

value_loss = (
    values
    - returns
).pow(2).mean()
```

---

# 13. 为什么 `values.detach()`？

这里：

```python
advantages = returns - values.detach()
```

非常重要。

Policy Loss 的目标是更新：

$$
\theta
$$

即 Policy Network。

Value Network 应该通过：

$$
L_{\text{value}}
$$

单独训练。

如果不 `.detach()`：

```text
Policy Loss
    ↓
Advantage
    ↓
Value Network
```

Policy Loss 的梯度也会流入 Value Network。

因此：

```python
values.detach()
```

表示：

> 在计算 Policy Loss 时，把 Value 当成固定数值，不允许 Policy Loss 更新 Value Network。

最终：

```text
Policy Loss
     ↓
Policy Network θ

Value Loss
     ↓
Value Network φ
```

---

# 14. 一个容易混淆的问题：Q 和 V 到底怎么来的？

理论上：

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
$$

但当前 REINFORCE + Baseline 并没有显式训练：

$$
Q(s,a)
$$

而是：

$$
\boxed{
G_t
\approx
Q^\pi(s_t,a_t)
}
$$

其中 \(G_t\) 是这一条 trajectory 真正采样得到的 Return。

Value：

$$
V_\phi(s_t)
$$

则是 Value Network 根据过去的训练数据学出来的预测。

因此：

```text
完整跑一个 Episode
        ↓
获得 r0,r1,...,rT
        ↓
计算 G0,G1,...,GT
        ↓
Gt ≈ 当前动作 Q(st,at) 的一次 MC 样本
        ↓
Value Network 预测 V(st)
        ↓
A_t = G_t - V(st)
```

---

# 15. 当前 Policy 到底是哪一个？

Value 定义：

$$
V^\pi(s)
=
\mathbb E_\pi[G_t|S_t=s]
$$

这里的：

$$
\pi
$$

就是当前正在训练的：

$$
\boxed{
\pi_\theta
}
$$

即 Policy Network。

例如同一个状态 \(s\)，当前策略：

$$
\pi_\theta(a|s)=[0.2,0.8]
$$

从这个状态反复采样运行，可能得到：

```text
第 1 次：G = 120
第 2 次：G = 80
第 3 次：G = 105
第 4 次：G = 95
```

那么：

$$
V^\pi(s)
$$

更接近这些结果的平均值。

而某一次：

$$
G_t=120
$$

只是一次 Monte Carlo 样本。

### 与 Task 2 的区别

Task 2：

```text
固定 Policy
↓
Bellman Policy Evaluation
↓
计算 Vπ(s)
```

Task 5：

```text
Policy Network 一直变化
↓
环境无法完整枚举
↓
Value Network 从采样数据中学习
↓
逼近 V^{πθ}(s)
```

定义没有变化，区别主要在求解方式。

---

# 16. Baseline 为什么不会改变 Policy Gradient 的期望？

考虑：

$$
\mathbb E
\left[
b(s)
\nabla_\theta
\log\pi_\theta(a|s)
\right]
$$

固定状态 \(s\)：

$$
\sum_a
\pi_\theta(a|s)
b(s)
\nabla_\theta
\log\pi_\theta(a|s)
$$

因为 \(b(s)\) 不依赖 action：

$$
=
b(s)
\sum_a
\pi_\theta(a|s)
\nabla_\theta
\log\pi_\theta(a|s)
$$

利用：

$$
\pi_\theta(a|s)
\nabla_\theta
\log\pi_\theta(a|s)
=
\nabla_\theta
\pi_\theta(a|s)
$$

得到：

$$
=
b(s)
\nabla_\theta
\sum_a
\pi_\theta(a|s)
$$

又因为：

$$
\sum_a\pi_\theta(a|s)=1
$$

所以：

$$
=
b(s)\nabla_\theta1
=
0
$$

因此减掉 Baseline 不改变梯度期望，但通常能够降低方差。

---

# 17. 为什么加入 Advantage 后 Episode Return 还是很抖？

这是本 Task 中一个非常重要的问题。

Baseline 主要降低的是：

$$
\boxed{
\operatorname{Var}
\left[
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
}
$$

即：

> Policy Gradient 估计器的方差。

它并不是直接降低：

$$
\operatorname{Var}(\text{Episode Return})
$$

训练时仍然：

```python
action = dist.sample()
```

Policy 是随机策略。

即使：

$$
\pi(a|s)=[0.02,0.98]
$$

仍可能有 2% 概率采到较差动作。

CartPole 又非常敏感，因此可能出现：

```text
500
500
495
220
500
480
...
```

这是正常现象。

另外：

$$
\hat A_t
=
G_t-V_\phi(s_t)
$$

中的：

$$
G_t
$$

仍然是 Monte Carlo Return，因此 REINFORCE + Baseline 仍然具有 MC 的高方差特性。

Baseline 的作用是：

```text
High Variance
      ↓
Lower Variance
```

而不是：

```text
High Variance
      ↓
Zero Variance
```

---

# 18. CartPole 实验

环境：

```text
CartPole-v1
```

状态：

$$
s=
[x,\dot{x},\theta,\dot{\theta}]
$$

包括：

- 小车位置；
- 小车速度；
- 杆子角度；
- 杆子角速度。

动作：

```text
0 → 向左推
1 → 向右推
```

Episode 最大 Return：

$$
500
$$

---

# 19. REINFORCE + Baseline 单次训练结果

1000 Episode 训练后，Moving Average 最终接近：

$$
480\sim500
$$

说明 Policy 已基本学会 CartPole。

训练后期仍会偶尔从 500 掉到 200 左右，这是随机 Policy 采样造成的正常现象。

Policy Loss 可能正负波动：

```text
21.7140
-6.8505
-1.2010
5.6989
...
```

这是正常的。

因为：

$$
A_t
$$

既可能为正，也可能为负。

因此 Policy Loss 不能像普通监督学习 Loss 一样简单理解成“越接近 0 越好”。

Policy Gradient 最核心的性能指标仍然是：

$$
\boxed{\text{Episode Return}}
$$

---

# 20. 最终实验：Vanilla REINFORCE vs REINFORCE + Baseline

为了避免单个 seed 的随机性，实验使用：

```python
SEEDS = [0, 1, 2, 3, 4]
```

每个算法训练：

$$
1000
$$

个 Episode。

比较：

1. Mean Learning Curve；
2. 不同 seed 的标准差；
3. 最终 100 Episode 平均 Return；
4. 达到 Return = 400 所需 Episode。

---

## 20.1 最终性能

实验结果：

| 方法 | 最后 100 Episode Return |
|---|---:|
| Vanilla REINFORCE | \(468.12\pm21.81\) |
| REINFORCE + Baseline | \(\mathbf{490.41\pm6.12}\) |

Baseline 的最终性能更高，同时不同 seed 之间的标准差明显更小。

标准差：

$$
21.81
\rightarrow
6.12
$$

下降约：

$$
72\%
$$

说明不同随机初始化下，Baseline 训练结果更加一致。

---

## 20.2 Sample Efficiency

阈值：

$$
\text{Moving Average Return}=400
$$

Vanilla：

```text
[613, 522, 553, 558, 500]
```

平均：

$$
549.2
$$

Baseline：

```text
[266, 213, 199, 264, 288]
```

平均：

$$
246.0
$$

即：

$$
549.2
\rightarrow
246.0
$$

达到相同性能时，Baseline 只需要约：

$$
\frac{246}{549.2}
\approx44.8\%
$$

的训练 Episode。

学习速度可以粗略理解为：

$$
\boxed{
2.23\times
}
$$

---

# 21. 实验结论

实验非常清晰地验证了：

$$
\boxed{
\text{Vanilla REINFORCE 直接使用 Monte Carlo Return，梯度估计方差较大。}
}
$$

加入 Value Baseline：

$$
\hat A_t
=
G_t-V_\phi(s_t)
$$

以后：

- 不改变策略梯度的期望；
- 降低梯度估计的方差；
- 显著提高 Sample Efficiency；
- 不同 seed 间更加稳定；
- 最终性能更高。

但它仍然属于：

$$
\boxed{
\text{Monte Carlo Policy Gradient}
}
$$

因此仍然需要完整 Episode，且仍存在一定高方差问题。

---

# 22. 本 Task 中重点问题总结

以下是本任务学习过程中重点讨论过的问题。

---

## Q1：Policy Gradient 和 DQN 到底区别在哪？

DQN：

$$
Q_\theta(s,a)
$$

先学习动作价值，再通过：

$$
\arg\max_aQ(s,a)
$$

得到策略。

Policy Gradient：

$$
\pi_\theta(a|s)
$$

直接学习动作概率。

---

## Q2：Policy Network 为什么不用 argmax？

因为训练阶段需要：

$$
a_t\sim\pi_\theta(a|s_t)
$$

通过 stochastic policy 产生 trajectory。

随机策略本身就具有探索能力。

---

## Q3：为什么代码中是 `-log_prob * G`？

理论上最大化：

$$
J(\theta)
$$

属于 Gradient Ascent。

PyTorch optimizer 默认做 Gradient Descent。

因此定义：

$$
L=-J
$$

所以：

```python
loss = -log_prob * G
```

---

## Q4：为什么 REINFORCE 要完整跑完一个 Episode 才更新？

因为：

$$
G_t
=
r_t+\gamma r_{t+1}+\cdots
$$

在时间步 \(t\) 时，未来 reward 还没有发生。

所以必须：

```text
完整 Episode
↓
获得全部 rewards
↓
计算 G_t
↓
更新 Policy
```

---

## Q5：Q 和 V 都是这一条 Episode 实际得到的吗？

不是。

在 REINFORCE + Baseline 中：

$$
G_t
$$

是当前 trajectory 真正采样得到的 Return，可以看作：

$$
Q^\pi(s_t,a_t)
$$

的一次 Monte Carlo 样本。

而：

$$
V_\phi(s_t)
$$

是 Value Network 预测的当前状态平均价值。

没有显式训练 Q Network。

---

## Q6：既然完整跑完一条 Episode，为什么还要 Value Network？

因为：

$$
G_t
$$

只是一次随机采样结果。

而：

$$
V^\pi(s)
$$

表示：

> 从状态 \(s\) 出发，按照当前 Policy 反复运行时，平均能够得到多少 Return。

所以：

$$
G_t-V(s_t)
$$

判断的不是“这一次奖励是不是正数”，而是：

> 这一次表现比当前 Policy 的正常水平好多少。

---

## Q7：这里的 \(V^\pi(s)\) 中的 \(\pi\) 是哪个 Policy？

就是当前正在训练的：

$$
\pi_\theta
$$

Policy 每个 Episode 都可能更新，因此 Value Network 实际上不断追踪：

$$
V^{\pi_{\theta_{\text{current}}}}(s)
$$

---

## Q8：Task 2 的 V 和这里的 V 有区别吗？

定义没有区别：

$$
V^\pi(s)
=
\mathbb E_\pi[G_t|S_t=s]
$$

区别在实现方式。

Task 2：

```text
固定 Policy
+
已知 MDP
+
Bellman Iteration
```

Task 5：

```text
Policy Network 不断变化
+
环境不可完整枚举
+
Value Network 从 trajectory 学习
```

---

## Q9：加入 Advantage 以后为什么训练曲线还是很抖？

因为 Advantage 降低的是：

$$
\text{Policy Gradient estimator variance}
$$

而不是直接消除：

$$
\text{Episode Return variance}
$$

训练阶段 Policy 仍然是 stochastic policy，而且 \(G_t\) 仍是 Monte Carlo Return，所以 Episode Return 仍可能大幅波动。

---

## Q10：怎么真正判断 Baseline 是否降低方差？

不能只看某一个 seed 的单 Episode Return 曲线。

更合理的方法是：

```text
多个随机 seed
↓
比较平均学习曲线
↓
比较 seed 间标准差
↓
比较 sample efficiency
```

最终实验显示：

```text
Vanilla:
468.12 ± 21.81

Baseline:
490.41 ± 6.12
```

以及：

```text
达到 Return = 400

Vanilla:
549.2 Episodes

Baseline:
246.0 Episodes
```

这才是 Baseline 改善训练稳定性的直接实验依据。

---

# 23. Task 05 最核心的公式

### Policy

$$
\boxed{
\pi_\theta(a|s)
}
$$

### Objective

$$
\boxed{
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
[R(\tau)]
}
$$

### Policy Gradient

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E
\left[
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
}
$$

### Return

$$
\boxed{
G_t
=
\sum_{k=t}^{T-1}
\gamma^{k-t}r_k
}
$$

### REINFORCE Loss

$$
\boxed{
L_{\text{policy}}
=
-\sum_t
G_t
\log\pi_\theta(a_t|s_t)
}
$$

### Value

$$
\boxed{
V^\pi(s)
=
\mathbb E_\pi[G_t|S_t=s]
}
$$

### Advantage

$$
\boxed{
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
}
$$

### Monte Carlo Advantage Estimate

$$
\boxed{
\hat A_t
=
G_t-V_\phi(s_t)
}
$$

### Policy Loss with Baseline

$$
\boxed{
L_{\text{policy}}
=
-\frac1T
\sum_t
\hat A_t
\log\pi_\theta(a_t|s_t)
}
$$

### Value Loss

$$
\boxed{
L_{\text{value}}
=
\frac1T
\sum_t
(V_\phi(s_t)-G_t)^2
}
$$

---

# 24. Task 05 知识链

```text
Task 3
Monte Carlo
   │
   ├── Return G_t
   │
   ↓
Task 4
Neural Network
   │
   ↓
Task 5
Policy Network
   │
   ↓
Policy Gradient
   │
   ↓
REINFORCE
   │
   ├── Monte Carlo Return
   │
   └── High Variance
   │
   ↓
Value Baseline
   │
   ↓
Advantage
   │
   ↓
REINFORCE + Baseline
```

最终可以记成：

$$
\boxed{
\text{REINFORCE}
=
\text{Monte Carlo}
+
\text{Policy Gradient}
}
$$

以及：

$$
\boxed{
\text{REINFORCE + Baseline}
=
\text{Policy Gradient}
+
\text{Value Function}
+
\text{Monte Carlo Advantage}
}
$$

---

# 25. 与下一 Task 的衔接

当前最大的遗留问题：

$$
G_t
$$

仍然必须等待完整 Episode 才能得到。

而且：

$$
G_t
$$

仍是 Monte Carlo 估计，方差仍然较大。

因此下一步自然是重新引入 Task 3 学过的：

$$
\boxed{\text{TD + Bootstrapping}}
$$

从：

$$
G_t-V(s_t)
$$

进一步变成：

$$
\boxed{
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
}
$$

这就是 TD Error。

因此下一阶段：

$$
\boxed{
\text{Policy Gradient}
+
\text{TD Learning}
=
\text{Actor-Critic}
}
$$

下一任务建议：

> **Task 06：Actor-Critic**

后续将重点学习：

- Actor 与 Critic 的职责；
- TD Error；
- Advantage 的在线估计；
- Actor-Critic 的更新方式；
- 从 Actor-Critic 进一步过渡到 A2C / GAE / PPO。

---

## Task 05 完成状态

- [x] 理解 Value-based 与 Policy-based 的区别
- [x] 理解 stochastic policy
- [x] 理解 Policy Gradient 核心公式
- [x] 理解 log-derivative trick
- [x] 从零实现 Policy Network
- [x] 实现 Vanilla REINFORCE
- [x] 理解 Monte Carlo Return
- [x] 理解 REINFORCE 高方差问题
- [x] 理解 Baseline
- [x] 理解 Advantage
- [x] 实现 Value Network
- [x] 实现 REINFORCE + Baseline
- [x] 理解 `.detach()` 的作用
- [x] 完成 CartPole 单算法实验
- [x] 完成 5-seed Vanilla vs Baseline 对比实验
- [x] 验证 Baseline 的样本效率与稳定性优势

**Task 05：完成。**


---

## Task 05 实验记录

至少使用 5 个 seed，比较 vanilla REINFORCE 与 baseline 的最终回报均值、标准差、达到阈值的 episode 数，并解释 baseline 为什么只改变方差而不改变期望梯度。

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
