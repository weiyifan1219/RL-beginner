# 任务二：MDP 与动态规划代码 Review 提示词

请对照 README 和 `tests/` 审查我的实现，优先检查：

1. 每个 `(s,a)` 的转移概率是否校验为 1，next state/reward/terminated 是否严格合法。
2. Bellman target 是否仅在 `terminated=False` 时加入 `gamma*V(next_state)`。
3. Policy evaluation 是否真的对动作概率和转移概率做了两层期望。
4. 每个 sweep 是同步还是原地更新，代码与说明是否一致。
5. Policy improvement 是否保留所有并列最大动作并形成和为 1 的分布。
6. `theta`、`max_iterations` 和 `converged` 是否能显式报告不收敛。
7. GridWorld 的撞墙、进入终点奖励、终点吸收语义是否与 golden values 一致。

不要直接重写全部代码。先给一个最高优先级、可用最小 MDP 复现的问题，列出预期与实际 backup，再给最小修复建议。
