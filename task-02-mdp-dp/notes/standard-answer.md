# Task 02 标准答案讲义

## 1. 从 return 到 Bellman 方程

折扣回报为：

```text
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
```

把第一项拆出来：

```text
G_t = R_{t+1} + γG_{t+1}
```

对状态 `s` 条件期望，并把策略的动作随机性、环境的转移随机性展开，就得到 Bellman expectation equation。它不是额外假设，而是 return 定义和 Markov 性质的直接结果。

## 2. 代码中的两层期望

`TabularMDP.expected_action_return(s,a,V,gamma)` 计算：

```text
Σ_{s',r} p(s',r|s,a) [r + γ(1-terminal)V(s')]
```

`bellman_expectation_backup` 再乘策略概率并对动作求和：

```text
Σ_a π(a|s) Q_backup(s,a)
```

拆成两层的好处是同一个 action lookahead 可供 policy evaluation、greedy improvement 和 value iteration 复用。

## 3. 为什么迭代会得到固定点

当 `γ<1` 时，Bellman expectation operator 是 contraction：两个价值函数经过一次 operator 后，最大范数距离至多缩小为原来的 `γ` 倍，因此有唯一固定点并收敛。

默认 GridWorld 使用 `γ=1`，但它是 episodic stochastic-shortest-path 问题；均匀随机策略会以概率 1 到达终点，仍有有限价值。标准答案会验证待评估策略几乎必然终止，否则拒绝一个可能非唯一的 Bellman fixed point。对值迭代，本课程保守支持每个状态存在终止路径、奖励非正且从零初始化的 shortest-path 子类；一般 continuing undiscounted MDP 不在这里静默求解。

## 4. 策略迭代

1. 初始化任意合法策略。
2. Policy evaluation：求当前 `V^π`。
3. Policy improvement：令新策略只支持数值上真正达到 `argmax_a Q^π(s,a)` 的动作；
   “非常接近”但仍更小的动作不算并列最优。
4. 若策略不变则结束，否则回到第 2 步。

有限 MDP 中，每次真正改变都不会让策略变差，因此最终到达最优策略。并列动作均分概率并不影响最优价值。

## 5. 值迭代

值迭代直接应用：

```text
V_{k+1}(s) = max_a Σ p(s',r|s,a)[r + γ(1-terminal)V_k(s')]
```

它相当于每次只做一个 sweep 的策略评估就立即改进。收敛后再做一次 greedy improvement 导出策略。

## 6. 同步和原地更新

标准答案每个 sweep 创建 `updated`，所有新值都基于上一轮完整 `values`，属于同步更新。原地更新（Gauss-Seidel 风格）常常更快，但状态遍历顺序会影响中间轨迹。教学和 golden tests 选择同步版本，公式映射最直接。

## 7. Terminal mask 的语义

对终止转移：

```text
target = reward
```

对非终止转移：

```text
target = reward + gamma * V(next_state)
```

即使终止状态数组槽里碰巧有 1000，也不应污染 target。Task 03 起还会遇到 Gymnasium 的 `truncated`：时间上限截断通常仍应 bootstrap，它与真正 `terminated` 不同。

## 8. Golden values 的意义

“函数返回且 converged=True”只能说明数值变化小，不能证明：

- 动作方向没有写反；
- 撞墙是否正确原地不动；
- 进入终点奖励是否为 -1；
- 终点是否正确 mask；
- 策略概率是否求和正确。

经典 4×4 价值表同时约束了这些语义，因此是高价值集成测试。

## 9. 实验解读

最优价值恰好是到最近终点的负 Manhattan 步数；并列最短路径对应多个箭头。随机策略价值远低于最优策略，因为它会撞墙、绕路甚至暂时远离终点。

修改 `theta`：

- 更小：更多 sweeps，数值更精确；
- 更大：更快停止，但策略未必改变，因为动作 gap 可能远大于数值误差；
- 不能只看 iterations，要同时看 `delta` 和最终策略。

## 10. 练习参考答案

1. **为什么 terminal state 仍保留动作行？** 统一 `(S,A)` 数组 shape；它们是吸收自环且立即终止，不影响此前目标。
2. **`γ=0` 学什么？** 只最大化即时奖励，完全忽略后继状态。
3. **policy iteration 一定比 value iteration 快吗？** 不能只比 outer iterations；一次完整 policy evaluation 包含许多 sweeps，应比较总 backup 次数和 wall time。
4. **为什么策略用概率矩阵而非动作向量？** 能同时表示随机策略、确定策略和并列最优策略，也为后续 on-policy 方法做准备。
5. **DP 最大限制是什么？** 需要完整环境模型且枚举全部状态；Task 03 将改用样本经验学习。
