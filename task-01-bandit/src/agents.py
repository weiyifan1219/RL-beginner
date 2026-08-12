"""Bandit 动作选择策略的透明参考实现。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def incremental_update(
    estimate: float,
    reward: float,
    count: int,
    step_size: float | None = None,
) -> float:
    """执行 ``Q <- Q + alpha * (R - Q)``。

    Args:
        estimate: 更新前的动作价值估计。
        reward: 本次观测奖励。
        count: 包含本次观测在内，该动作被选择的次数，必须 >= 1。
        step_size: ``None`` 表示样本平均 ``alpha=1/count``；否则使用固定步长。
    """
    if count < 1:
        raise ValueError("count must be positive")
    if step_size is None:
        alpha = 1.0 / count
    else:
        if not np.isfinite(step_size) or not 0.0 < step_size <= 1.0:
            raise ValueError("step_size must lie in (0, 1]")
        alpha = float(step_size)
    return float(estimate + alpha * (reward - estimate))


def _validate_n_arms(n_arms: int) -> int:
    if isinstance(n_arms, bool) or not isinstance(n_arms, (int, np.integer)) or n_arms < 1:
        raise ValueError("n_arms must be a positive integer")
    return int(n_arms)


def _random_argmax(values: FloatArray, rng: np.random.Generator) -> int:
    """在最大值并列时随机打破平局，避免永远偏向最小下标。"""
    winners = np.flatnonzero(np.isclose(values, np.max(values), rtol=1e-12, atol=1e-12))
    return int(rng.choice(winners))


class EpsilonGreedyAgent:
    """ε-greedy 动作价值算法。

    ``estimates`` 与 ``counts`` 均为形状 ``(n_arms,)`` 的公开数组，便于教学观察。
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        step_size: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_arms = _validate_n_arms(n_arms)
        if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if not np.isfinite(initial_value):
            raise ValueError("initial_value must be finite")
        if step_size is not None and (not np.isfinite(step_size) or not 0.0 < step_size <= 1.0):
            raise ValueError("step_size must lie in (0, 1]")
        self.epsilon = float(epsilon)
        self.step_size = None if step_size is None else float(step_size)
        self.estimates: FloatArray = np.full(self.n_arms, initial_value, dtype=np.float64)
        self.counts: IntArray = np.zeros(self.n_arms, dtype=np.int64)
        self._rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_arms))
        return _random_argmax(self.estimates, self._rng)

    def update(self, action: int, reward: float) -> None:
        action = self._validate_action(action)
        self.counts[action] += 1
        self.estimates[action] = incremental_update(
            self.estimates[action], reward, int(self.counts[action]), self.step_size
        )

    def _validate_action(self, action: int) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action = int(action)
        if not 0 <= action < self.n_arms:
            raise IndexError(f"action must be in [0, {self.n_arms}), got {action}")
        return action


class UCBAgent(EpsilonGreedyAgent):
    """Upper Confidence Bound 动作选择。

    每个尚未尝试的动作会先被选择一次；随后最大化
    ``Q(a) + c * sqrt(log(t) / N(a))``。
    """

    def __init__(
        self,
        n_arms: int,
        c: float = 2.0,
        initial_value: float = 0.0,
        step_size: float | None = None,
        seed: int | None = None,
    ) -> None:
        if not np.isfinite(c) or c < 0:
            raise ValueError("c must be a non-negative finite number")
        super().__init__(n_arms, epsilon=0.0, initial_value=initial_value, step_size=step_size, seed=seed)
        self.c = float(c)

    def select_action(self) -> int:
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(self._rng.choice(untried))
        total_steps = int(np.sum(self.counts))
        bonuses = self.c * np.sqrt(np.log(total_steps) / self.counts)
        return _random_argmax(self.estimates + bonuses, self._rng)


class ThompsonSamplingAgent:
    """Beta-Bernoulli Thompson Sampling。

    该共轭实现只适用于奖励严格为 0/1 的 Bernoulli bandit。对 Gaussian 奖励需要
    不同的似然与先验，不能直接复用本类。
    """

    def __init__(
        self,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.n_arms = _validate_n_arms(n_arms)
        if not np.isfinite(alpha_prior) or alpha_prior <= 0:
            raise ValueError("alpha_prior must be positive and finite")
        if not np.isfinite(beta_prior) or beta_prior <= 0:
            raise ValueError("beta_prior must be positive and finite")
        self.alphas: FloatArray = np.full(self.n_arms, alpha_prior, dtype=np.float64)
        self.betas: FloatArray = np.full(self.n_arms, beta_prior, dtype=np.float64)
        self.counts: IntArray = np.zeros(self.n_arms, dtype=np.int64)
        self._rng = np.random.default_rng(seed)

    @property
    def estimates(self) -> FloatArray:
        """Beta 后验均值 ``alpha / (alpha + beta)``。"""
        return self.alphas / (self.alphas + self.betas)

    def select_action(self) -> int:
        samples = self._rng.beta(self.alphas, self.betas)
        return _random_argmax(samples, self._rng)

    def update(self, action: int, reward: float) -> None:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action = int(action)
        if not 0 <= action < self.n_arms:
            raise IndexError(f"action must be in [0, {self.n_arms}), got {action}")
        if reward not in (0, 0.0, 1, 1.0):
            raise ValueError("Beta-Bernoulli Thompson Sampling requires Bernoulli rewards 0 or 1")
        self.alphas[action] += float(reward)
        self.betas[action] += 1.0 - float(reward)
        self.counts[action] += 1
