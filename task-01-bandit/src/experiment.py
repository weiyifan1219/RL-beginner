"""可复现的 bandit 单次轨迹和多随机种子实验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class Bandit(Protocol):
    n_arms: int

    @property
    def optimal_arm(self) -> int: ...

    def pull(self, action: int) -> float: ...

    def regret(self, action: int) -> float: ...


class BanditAgent(Protocol):
    n_arms: int

    def select_action(self) -> int: ...

    def update(self, action: int, reward: float) -> None: ...


@dataclass(frozen=True)
class EpisodeResult:
    """一次 bandit 轨迹；所有数组形状均为 ``(n_steps,)``。"""

    actions: IntArray
    rewards: FloatArray
    optimal_actions: BoolArray
    instantaneous_regret: FloatArray
    cumulative_regret: FloatArray


@dataclass(frozen=True)
class ExperimentResult:
    """对 ``n_runs`` 次独立轨迹逐时间步聚合后的结果。"""

    n_runs: int
    n_steps: int
    mean_reward: FloatArray
    reward_standard_error: FloatArray
    optimal_action_rate: FloatArray
    mean_instantaneous_regret: FloatArray
    mean_cumulative_regret: FloatArray

    def summary(self) -> dict[str, float | int]:
        tail = max(1, self.n_steps // 10)
        return {
            "n_runs": self.n_runs,
            "n_steps": self.n_steps,
            "mean_reward": float(np.mean(self.mean_reward)),
            "optimal_action_rate_last_10_percent": float(np.mean(self.optimal_action_rate[-tail:])),
            "final_cumulative_regret": float(self.mean_cumulative_regret[-1]),
        }


def _validate_budget(n_steps: int, n_runs: int | None = None) -> None:
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if n_runs is not None and (
        isinstance(n_runs, bool) or not isinstance(n_runs, (int, np.integer)) or n_runs < 1
    ):
        raise ValueError("n_runs must be a positive integer")


def run_episode(env: Bandit, agent: BanditAgent, n_steps: int) -> EpisodeResult:
    """运行一次交互。

    Regret 在 ``pull`` 之前计算，因为非平稳环境会在 ``pull`` 后改变真实动作价值。
    """
    _validate_budget(n_steps)
    if env.n_arms != agent.n_arms:
        raise ValueError(f"env has {env.n_arms} arms but agent has {agent.n_arms}")

    actions = np.empty(n_steps, dtype=np.int64)
    rewards = np.empty(n_steps, dtype=np.float64)
    optimal_actions = np.empty(n_steps, dtype=np.bool_)
    instantaneous_regret = np.empty(n_steps, dtype=np.float64)

    for step in range(n_steps):
        action = agent.select_action()
        regret = env.regret(action)
        instantaneous_regret[step] = regret
        # 并列最优臂的下标可能不同，但只要期望 gap 为 0 都应计为最优动作。
        optimal_actions[step] = regret == 0.0
        reward = env.pull(action)
        agent.update(action, reward)
        actions[step] = action
        rewards[step] = reward

    return EpisodeResult(
        actions=actions,
        rewards=rewards,
        optimal_actions=optimal_actions,
        instantaneous_regret=instantaneous_regret,
        cumulative_regret=np.cumsum(instantaneous_regret),
    )


def run_experiment(
    env_factory: Callable[[int], Bandit],
    agent_factory: Callable[[int], BanditAgent],
    n_runs: int,
    n_steps: int,
    seed: int = 0,
) -> ExperimentResult:
    """运行独立重复实验并逐时间步聚合。

    ``env_factory(seed)`` 与 ``agent_factory(seed)`` 必须每次返回全新对象。环境和 agent
    使用从同一 ``SeedSequence`` 派生但互不相同的 seed。
    """
    _validate_budget(n_steps, n_runs)
    children = np.random.SeedSequence(seed).spawn(2 * n_runs)
    rewards = np.empty((n_runs, n_steps), dtype=np.float64)
    optimal_actions = np.empty((n_runs, n_steps), dtype=np.float64)
    regrets = np.empty((n_runs, n_steps), dtype=np.float64)

    for run in range(n_runs):
        env_seed = int(children[2 * run].generate_state(1, dtype=np.uint32)[0])
        agent_seed = int(children[2 * run + 1].generate_state(1, dtype=np.uint32)[0])
        episode = run_episode(env_factory(env_seed), agent_factory(agent_seed), n_steps)
        rewards[run] = episode.rewards
        optimal_actions[run] = episode.optimal_actions
        regrets[run] = episode.instantaneous_regret

    if n_runs > 1:
        reward_standard_error = np.std(rewards, axis=0, ddof=1) / np.sqrt(n_runs)
    else:
        # 一个观测无法估计跨 run 的标准误；NaN 比误导性的 0 更诚实。
        reward_standard_error = np.full(n_steps, np.nan, dtype=np.float64)
    return ExperimentResult(
        n_runs=n_runs,
        n_steps=n_steps,
        mean_reward=np.mean(rewards, axis=0),
        reward_standard_error=reward_standard_error,
        optimal_action_rate=np.mean(optimal_actions, axis=0),
        mean_instantaneous_regret=np.mean(regrets, axis=0),
        mean_cumulative_regret=np.mean(np.cumsum(regrets, axis=1), axis=0),
    )


def plot_results(results: dict[str, ExperimentResult], output_path: str | Path | None = None):
    """绘制平均奖励、最优动作率和累计 pseudo-regret，返回 Matplotlib Figure。"""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("results must not be empty")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, result in results.items():
        steps = np.arange(1, result.n_steps + 1)
        axes[0].plot(steps, result.mean_reward, label=name)
        axes[1].plot(steps, result.optimal_action_rate, label=name)
        axes[2].plot(steps, result.mean_cumulative_regret, label=name)
    axes[0].set(title="Average reward", xlabel="Step", ylabel="Reward")
    axes[1].set(title="Optimal action rate", xlabel="Step", ylabel="Rate", ylim=(0.0, 1.05))
    axes[2].set(title="Cumulative pseudo-regret", xlabel="Step", ylabel="Regret")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
    return figure
