"""多臂老虎机环境。

环境只负责产生奖励与暴露真实期望奖励。算法不会读取真实期望；它只在实验代码中
用于计算 pseudo-regret（期望遗憾）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _as_nonempty_finite_vector(values: Sequence[float], name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


class GaussianBandit:
    """平稳 Gaussian k 臂老虎机。

    Args:
        means: 每个动作的真实期望奖励，形状 ``(n_arms,)``。
        std: 所有动作共享的奖励标准差，必须非负；0 可构造确定性教学环境。
        seed: NumPy 随机种子。

    ``pull(action)`` 返回一个 Python ``float``，服从
    :math:`N(\text{means[action]}, \text{std}^2)`。
    """

    def __init__(self, means: Sequence[float], std: float = 1.0, seed: int | None = None) -> None:
        self._means = _as_nonempty_finite_vector(means, "means")
        if not np.isfinite(std) or std < 0:
            raise ValueError("std must be a non-negative finite number")
        self.std = float(std)
        self._rng = np.random.default_rng(seed)

    @property
    def n_arms(self) -> int:
        return int(self._means.size)

    @property
    def expected_rewards(self) -> FloatArray:
        """返回真实期望奖励的副本，避免调用者意外修改环境。"""
        return self._means.copy()

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self._means))

    @property
    def optimal_mean(self) -> float:
        return float(np.max(self._means))

    def pull(self, action: int) -> float:
        action = self._validate_action(action)
        return float(self._rng.normal(self._means[action], self.std))

    def regret(self, action: int) -> float:
        """返回动作的单步 pseudo-regret，而不是带噪声的 realized regret。"""
        action = self._validate_action(action)
        return self.optimal_mean - float(self._means[action])

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def _validate_action(self, action: int) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action = int(action)
        if not 0 <= action < self.n_arms:
            raise IndexError(f"action must be in [0, {self.n_arms}), got {action}")
        return action


class BernoulliBandit(GaussianBandit):
    """Bernoulli k 臂老虎机，每次奖励严格为 0.0 或 1.0。

    Args:
        probabilities: 每个动作成功并获得 1 的概率，形状 ``(n_arms,)``。
        seed: NumPy 随机种子。
    """

    def __init__(self, probabilities: Sequence[float], seed: int | None = None) -> None:
        probabilities_array = _as_nonempty_finite_vector(probabilities, "probabilities")
        if np.any((probabilities_array < 0.0) | (probabilities_array > 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")
        self._means = probabilities_array
        self.std = float("nan")
        self._rng = np.random.default_rng(seed)

    def pull(self, action: int) -> float:
        action = self._validate_action(action)
        return float(self._rng.random() < self._means[action])


class NonStationaryGaussianBandit(GaussianBandit):
    """动作真实价值按随机游走变化的 Gaussian bandit。

    奖励先按当前均值产生，随后所有动作价值独立漂移：
    ``q_{t+1}(a) = q_t(a) + Normal(0, random_walk_std)``。
    """

    def __init__(
        self,
        initial_means: Sequence[float],
        std: float = 1.0,
        random_walk_std: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__(initial_means, std=std, seed=seed)
        if not np.isfinite(random_walk_std) or random_walk_std <= 0:
            raise ValueError("random_walk_std must be a positive finite number")
        self.random_walk_std = float(random_walk_std)
        self._initial_means = self._means.copy()

    def pull(self, action: int) -> float:
        action = self._validate_action(action)
        reward = float(self._rng.normal(self._means[action], self.std))
        self._means += self._rng.normal(0.0, self.random_walk_std, size=self.n_arms)
        return reward

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        self._means = self._initial_means.copy()
