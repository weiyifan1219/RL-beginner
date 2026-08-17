"""Greedy agent."""

from __future__ import annotations

import numpy as np


class GreedyAgent:
    """
    纯 Greedy Agent。

    始终选择当前 Q(a) 最大的动作：

        A_t = argmax_a Q_t(a)

    Q(a) 使用 Sample Average 更新。
    """

    def __init__(
        self,
        n_actions: int,
        seed: int | None = None,
    ):
        self.n_actions = n_actions

        # 动作价值估计
        self.Q = np.zeros(
            n_actions,
            dtype=np.float64,
        )

        # 每个动作被选择的次数
        self.N = np.zeros(
            n_actions,
            dtype=np.int64,
        )

        self.rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        """
        选择当前 Q 最大的动作。

        如果多个动作并列最大，
        随机打破平局，避免偏向较小下标。
        """

        max_q = np.max(self.Q)

        best_actions = np.flatnonzero(
            np.isclose(
                self.Q,
                max_q,
            )
        )

        return int(
            self.rng.choice(best_actions)
        )

    def update(
        self,
        action: int,
        reward: float,
    ) -> None:
        """
        Sample Average Update:

        Q(a) <- Q(a)
                + 1/N(a) * [R - Q(a)]
        """

        self.N[action] += 1

        alpha = 1.0 / self.N[action]

        self.Q[action] += (
            alpha
            * (
                reward
                - self.Q[action]
            )
        )