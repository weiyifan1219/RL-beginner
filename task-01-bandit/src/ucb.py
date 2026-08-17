"""Upper Confidence Bound agent."""

from __future__ import annotations

import numpy as np


class UCBAgent:
    """
    Upper Confidence Bound (UCB) Agent。

    动作选择：

        A_t = argmax_a [
            Q(a)
            + c * sqrt(log(t) / N(a))
        ]

    第一项：
        当前价值估计

    第二项：
        Exploration Bonus
    """

    def __init__(
        self,
        n_actions: int,
        c: float = 1.0,
        seed: int | None = None,
    ):
        if c < 0:
            raise ValueError(
                "c must be non-negative"
            )

        self.n_actions = n_actions

        self.c = c

        self.Q = np.zeros(
            n_actions,
            dtype=np.float64,
        )

        self.N = np.zeros(
            n_actions,
            dtype=np.int64,
        )

        self.rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        """
        根据 UCB Score 选择动作。
        """

        # 没有探索过的动作优先尝试
        unvisited_actions = np.flatnonzero(
            self.N == 0
        )

        if len(unvisited_actions) > 0:

            return int(
                self.rng.choice(
                    unvisited_actions
                )
            )

        # 总交互次数
        t = np.sum(self.N)

        # Exploration Bonus
        bonus = (
            self.c
            * np.sqrt(
                np.log(t)
                / self.N
            )
        )

        ucb_values = self.Q + bonus

        max_ucb = np.max(
            ucb_values
        )

        best_actions = np.flatnonzero(
            np.isclose(
                ucb_values,
                max_ucb,
            )
        )

        return int(
            self.rng.choice(
                best_actions
            )
        )

    def update(
        self,
        action: int,
        reward: float,
    ) -> None:
        """
        Sample Average Update.
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