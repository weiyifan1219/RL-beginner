"""有限状态、有限动作的表格型 Markov Decision Process。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Transition:
    """一个离散转移分支。

    Attributes:
        probability: ``P(S'=next_state, R=reward | S, A)``。
        next_state: 后继状态编号。
        reward: 该转移产生的即时奖励。
        terminated: 是否到达真正的 MDP 终止状态；为真时不 bootstrap。
    """

    probability: float
    next_state: int
    reward: float
    terminated: bool


TransitionLike = Transition | tuple[float, int, float, bool]


class TabularMDP:
    """经过严格校验的离散表格型 MDP。

    Args:
        transitions: ``transitions[s][a]`` 是一个非空转移分支列表，每个分支为
            ``(probability, next_state, reward, terminated)``。所有状态动作数必须相同，
            每个 ``(s,a)`` 的 probability 之和必须为 1。
    """

    def __init__(self, transitions: Sequence[Sequence[Sequence[TransitionLike]]]) -> None:
        if not isinstance(transitions, Sequence) or len(transitions) == 0:
            raise ValueError("transition table must contain at least one state")
        self.n_states = len(transitions)
        first_actions = transitions[0]
        if not isinstance(first_actions, Sequence) or len(first_actions) == 0:
            raise ValueError("each state must contain at least one action")
        self.n_actions = len(first_actions)

        normalized_states: list[tuple[tuple[Transition, ...], ...]] = []
        for state, actions in enumerate(transitions):
            if not isinstance(actions, Sequence) or len(actions) != self.n_actions:
                raise ValueError("all states must have the same number of actions")
            normalized_actions: list[tuple[Transition, ...]] = []
            for action, outcomes in enumerate(actions):
                if not isinstance(outcomes, Sequence) or len(outcomes) == 0:
                    raise ValueError(f"transitions[{state}][{action}] must be non-empty")
                branches = tuple(self._normalize_transition(item) for item in outcomes)
                total_probability = sum(item.probability for item in branches)
                if not np.isclose(total_probability, 1.0, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"transition probabilities for state={state}, action={action} must sum to 1"
                    )
                normalized_actions.append(branches)
            normalized_states.append(tuple(normalized_actions))
        self.transitions = tuple(normalized_states)

    def _normalize_transition(self, item: TransitionLike) -> Transition:
        if isinstance(item, Transition):
            raw = (item.probability, item.next_state, item.reward, item.terminated)
        elif isinstance(item, tuple) and len(item) == 4:
            raw = item
        else:
            raise TypeError("each transition must be Transition or a four-item tuple")
        probability, next_state, reward, terminated = raw
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise TypeError("transition probability must be a real number")
        probability = float(probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("transition probability must lie in [0, 1]")
        if isinstance(next_state, bool) or not isinstance(next_state, Integral):
            raise TypeError("next_state must be an integer")
        next_state = int(next_state)
        if not 0 <= next_state < self.n_states:
            raise ValueError(f"next_state must be in [0, {self.n_states})")
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise TypeError("reward must be a real number")
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not isinstance(terminated, (bool, np.bool_)):
            raise TypeError("terminated must be boolean")
        return Transition(probability, next_state, reward, bool(terminated))

    def expected_action_return(
        self,
        state: int,
        action: int,
        values: NDArray[np.floating] | Sequence[float],
        gamma: float,
    ) -> float:
        """计算一个 ``(s,a)`` 的 Bellman one-step lookahead。

        ``sum p * [r + gamma * (1-terminated) * V(s')]``
        """
        state = self._validate_index(state, self.n_states, "state")
        action = self._validate_index(action, self.n_actions, "action")
        values_array = np.asarray(values, dtype=np.float64)
        if values_array.shape != (self.n_states,):
            raise ValueError(f"values must have shape ({self.n_states},)")
        if not np.all(np.isfinite(values_array)):
            raise ValueError("values must be finite")
        gamma = self._validate_gamma(gamma)
        total = 0.0
        for transition in self.transitions[state][action]:
            bootstrap = 0.0 if transition.terminated else gamma * values_array[transition.next_state]
            total += transition.probability * (transition.reward + bootstrap)
        return float(total)

    @staticmethod
    def _validate_index(value: int, upper: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        value = int(value)
        if not 0 <= value < upper:
            raise IndexError(f"{name} must be in [0, {upper}), got {value}")
        return value

    @staticmethod
    def _validate_gamma(gamma: float) -> float:
        if isinstance(gamma, bool) or not isinstance(gamma, Real):
            raise TypeError("gamma must be a real number")
        gamma = float(gamma)
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        return gamma
