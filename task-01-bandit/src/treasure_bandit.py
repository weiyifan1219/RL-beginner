"""Treasure Hunt Multi-Armed Bandit environment."""

from __future__ import annotations

import numpy as np


class TreasureHuntBandit:
    """
    寻宝机器人 Multi-Armed Bandit 环境。

    每个区域都有一个固定但对 Agent 隐藏的宝物发现概率 q*(a)。

    每次 Agent 选择一个区域：
        reward = 1: 找到宝物
        reward = 0: 没找到宝物

    每轮结束后机器人自动返回基地，因此没有状态转移。
    """

    def __init__(
        self,
        treasure_probs,
        seed: int | None = None,
    ):
        self.treasure_probs = np.asarray(
            treasure_probs,
            dtype=np.float64,
        )

        if self.treasure_probs.ndim != 1:
            raise ValueError("treasure_probs must be a 1D array")

        if len(self.treasure_probs) == 0:
            raise ValueError("treasure_probs cannot be empty")

        if np.any(
            (self.treasure_probs < 0)
            | (self.treasure_probs > 1)
        ):
            raise ValueError(
                "treasure probabilities must lie in [0, 1]"
            )

        self.n_actions = len(self.treasure_probs)

        self.rng = np.random.default_rng(seed)

    def step(self, action: int) -> float:
        """
        前往指定区域探索一次。

        Returns
        -------
        reward : float
            1.0 表示找到宝物；
            0.0 表示没有找到宝物。
        """

        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"action must be in [0, {self.n_actions})"
            )

        probability = self.treasure_probs[action]

        found_treasure = (
            self.rng.random() < probability
        )

        return float(found_treasure)

    @property
    def optimal_action(self) -> int:
        """
        返回真实最优区域。

        注意：
        这个信息只应该用于实验评价，
        Agent 本身不允许访问。
        """

        return int(
            np.argmax(self.treasure_probs)
        )

    @property
    def optimal_value(self) -> float:
        """
        最优动作的真实期望奖励。
        """

        return float(
            np.max(self.treasure_probs)
        )