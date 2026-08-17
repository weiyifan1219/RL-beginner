"""Epsilon-Greedy agent."""

from __future__ import annotations

import numpy as np


class EpsilonGreedyAgent:
    """
    ε-Greedy Agent。

    epsilon 概率：
        Exploration
        随机选择动作

    1 - epsilon 概率：
        Exploitation
        选择当前 Q 最大的动作
    """

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        if not 0 <= epsilon <= 1:
            raise ValueError(
                "epsilon must lie in [0, 1]"
            )

        self.n_actions = n_actions

        self.epsilon = epsilon

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
        ε-Greedy action selection.
        """

        # Exploration
        if self.rng.random() < self.epsilon:

            return int(
                self.rng.integers(
                    self.n_actions
                )
            )

        # Exploitation
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