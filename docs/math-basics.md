# 数学预备知识

| 主题 | 至少应掌握 | 对应 RL 用处 |
|---|---|---|
| 概率 | 条件概率、期望、方差、Bayes | 策略、环境转移、采样估计 |
| 线性代数 | 向量梯度、Jacobian、二次型 | 神经网络参数与高斯策略 |
| 微积分 | 链式法则、log 导数技巧 | policy gradient 推导 |
| 优化 | SGD/Adam、约束优化、KL | PPO 信赖域与温度系数 |
| Python/PyTorch | tensor shape、autograd、批处理 | 复现实验与定位数值问题 |

最重要的技巧：`∇θ E[f(x)] = E[f(x) ∇θ log pθ(x)]`。它使不可微环境中的策略优化成为可能。

