"""Beta-Bernoulli Thompson Sampling agent."""

from __future__ import annotations

import numpy as np


class ThompsonSamplingAgent:
    """
    Beta-Bernoulli Thompson Sampling。

    对每个动作维护：

        p(a) ~ Beta(alpha[a], beta[a])

    每一步从每个动作的后验分布采样，
    选择采样值最大的动作。
    """

    def __init__(
        self,
        n_actions: int,
        seed: int | None = None,
    ):
        self.n_actions = n_actions

        # Beta(1, 1) uniform prior
        self.alpha = np.ones(
            n_actions,
            dtype=np.float64,
        )

        self.beta = np.ones(
            n_actions,
            dtype=np.float64,
        )

        self.N = np.zeros(
            n_actions,
            dtype=np.int64,
        )

        self.rng = np.random.default_rng(seed)

    @property
    def Q(self):
        """
        Beta posterior mean:

            E[p]
            = alpha / (alpha + beta)

        主要为了和其他 Agent 使用统一接口。
        """

        return (
            self.alpha
            /
            (
                self.alpha
                + self.beta
            )
        )

    def select_action(self) -> int:
        """
        对每个动作从 Beta posterior 中采样。
        """

        samples = self.rng.beta(
            self.alpha,
            self.beta,
        )

        return int(
            np.argmax(samples)
        )

    def update(
        self,
        action: int,
        reward: float,
    ) -> None:
        """
        更新 Beta posterior。

        reward = 1:
            alpha += 1

        reward = 0:
            beta += 1
        """

        if reward not in (0.0, 1.0):
            raise ValueError(
                "Thompson Sampling requires "
                "Bernoulli rewards 0 or 1"
            )

        self.N[action] += 1

        self.alpha[action] += reward

        self.beta[action] += (
            1.0 - reward
        )