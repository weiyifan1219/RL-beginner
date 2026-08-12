"""Sutton & Barto 风格的确定性 4 邻域 GridWorld。"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .mdp import TabularMDP


ACTION_NAMES = ("up", "right", "down", "left")
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class GridWorld(TabularMDP):
    """矩形 GridWorld，两端角落为终止状态。

    非终止状态每走一步奖励 -1；撞墙会留在原地并仍获得 -1；进入终止角落后 episode
    结束。终止状态自身是吸收状态，后续奖励为 0。状态按 row-major 编号。
    """

    def __init__(self, rows: int = 4, cols: int = 4) -> None:
        self.rows = self._validate_dimension(rows, "rows")
        self.cols = self._validate_dimension(cols, "cols")
        n_states = self.rows * self.cols
        self.terminal_states = (0, n_states - 1)

        table: list[list[list[tuple[float, int, float, bool]]]] = []
        for state in range(n_states):
            state_actions: list[list[tuple[float, int, float, bool]]] = []
            if state in self.terminal_states:
                for _ in ACTION_NAMES:
                    state_actions.append([(1.0, state, 0.0, True)])
            else:
                row, col = divmod(state, self.cols)
                for row_delta, col_delta in ACTION_DELTAS:
                    next_row = min(max(row + row_delta, 0), self.rows - 1)
                    next_col = min(max(col + col_delta, 0), self.cols - 1)
                    next_state = next_row * self.cols + next_col
                    state_actions.append(
                        [(1.0, next_state, -1.0, next_state in self.terminal_states)]
                    )
            table.append(state_actions)
        super().__init__(table)

    @staticmethod
    def _validate_dimension(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
        return int(value)

    def state_to_coord(self, state: int) -> tuple[int, int]:
        state = self._validate_index(state, self.n_states, "state")
        return divmod(state, self.cols)

    def coord_to_state(self, coordinate: tuple[int, int]) -> int:
        if not isinstance(coordinate, tuple) or len(coordinate) != 2:
            raise TypeError("coordinate must be a (row, col) tuple")
        row, col = coordinate
        if (
            isinstance(row, bool)
            or isinstance(col, bool)
            or not isinstance(row, Integral)
            or not isinstance(col, Integral)
            or not 0 <= row < self.rows
            or not 0 <= col < self.cols
        ):
            raise IndexError("coordinate lies outside the grid")
        return int(row) * self.cols + int(col)

    def uniform_random_policy(self) -> NDArray[np.float64]:
        """返回形状 ``(n_states, 4)``、每行动作概率均为 1/4 的策略。"""
        return np.full((self.n_states, self.n_actions), 1.0 / self.n_actions, dtype=np.float64)
