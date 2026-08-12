"""GridWorld 价值函数和随机策略可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .gridworld import GridWorld


ARROWS = ("↑", "→", "↓", "←")


def policy_labels(env: GridWorld, policy: NDArray[np.floating]) -> list[str]:
    """将 ``(S,A)`` 策略变成每个状态的箭头字符串。"""
    array = np.asarray(policy, dtype=np.float64)
    if array.shape != (env.n_states, env.n_actions):
        raise ValueError(f"policy must have shape ({env.n_states}, {env.n_actions})")
    labels: list[str] = []
    for state, probabilities in enumerate(array):
        if state in env.terminal_states:
            labels.append("T")
            continue
        actions = np.flatnonzero(probabilities > 0.0)
        labels.append("".join(ARROWS[action] for action in actions))
    return labels


def plot_value_and_policy(
    env: GridWorld,
    values: Sequence[float],
    policy: NDArray[np.floating],
    output_path: str | Path | None = None,
):
    """返回含价值热图与策略箭头的 Matplotlib Figure。"""
    import matplotlib.pyplot as plt

    values_array = np.asarray(values, dtype=np.float64)
    if values_array.shape != (env.n_states,):
        raise ValueError(f"values must have shape ({env.n_states},)")
    labels = policy_labels(env, policy)
    grid = values_array.reshape(env.rows, env.cols)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    image = axes[0].imshow(grid, cmap="viridis")
    axes[0].set_title("State values")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(np.zeros_like(grid), cmap="Greys", vmin=0.0, vmax=1.0)
    axes[1].set_title("Greedy policy")
    for state, value in enumerate(values_array):
        row, col = env.state_to_coord(state)
        axes[0].text(col, row, f"{value:.1f}", ha="center", va="center", color="white")
        axes[1].text(col, row, labels[state], ha="center", va="center", fontsize=15)
    for axis in axes:
        axis.set_xticks(range(env.cols))
        axis.set_yticks(range(env.rows))
        axis.set_xlabel("column")
        axis.set_ylabel("row")
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
    return figure
