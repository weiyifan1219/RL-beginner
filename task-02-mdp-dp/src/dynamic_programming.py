"""策略评估、策略迭代和值迭代的 NumPy 标准答案。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .mdp import TabularMDP


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EvaluationResult:
    values: FloatArray
    iterations: int
    converged: bool
    delta: float


@dataclass(frozen=True)
class DPResult:
    values: FloatArray
    policy: FloatArray
    iterations: int
    converged: bool
    delta: float


def _validate_solver_args(gamma: float, theta: float, max_iterations: int) -> tuple[float, float, int]:
    gamma = TabularMDP._validate_gamma(gamma)
    if isinstance(theta, bool) or not isinstance(theta, Real):
        raise TypeError("theta must be a real number")
    theta = float(theta)
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be positive and finite")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    return gamma, theta, int(max_iterations)


def _validate_values(mdp: TabularMDP, values: Sequence[float] | FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (mdp.n_states,):
        raise ValueError(f"values must have shape ({mdp.n_states},)")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return array


def _validate_policy(mdp: TabularMDP, policy: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
    array = np.asarray(policy, dtype=np.float64)
    expected_shape = (mdp.n_states, mdp.n_actions)
    if array.shape != expected_shape:
        raise ValueError(f"policy must have shape {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("policy must be finite")
    if np.any(array < 0):
        raise ValueError("policy probabilities must be non-negative")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each policy row must sum to 1")
    return array


def _policy_terminates_almost_surely(mdp: TabularMDP, policy: FloatArray) -> bool:
    """以单调 fixed-point 迭代计算各状态最终终止的概率。"""
    termination = np.zeros(mdp.n_states, dtype=np.float64)
    for _ in range(100_000):
        updated = np.zeros_like(termination)
        for state in range(mdp.n_states):
            for action in range(mdp.n_actions):
                for transition in mdp.transitions[state][action]:
                    continuation = 1.0 if transition.terminated else termination[transition.next_state]
                    updated[state] += (
                        policy[state, action] * transition.probability * continuation
                    )
        if np.max(np.abs(updated - termination)) < 1e-13:
            termination = updated
            break
        termination = updated
    return bool(np.all(termination >= 1.0 - 1e-10))


def _validate_undiscounted_control_problem(
    mdp: TabularMDP,
    initial_values: Sequence[float] | FloatArray | None,
) -> None:
    """检查本课程支持的 gamma=1 stochastic-shortest-path 子类。"""
    reachable: set[int] = set()
    changed = True
    while changed:
        changed = False
        for state in range(mdp.n_states):
            if state in reachable:
                continue
            has_path = any(
                transition.probability > 0.0
                and (transition.terminated or transition.next_state in reachable)
                for action in range(mdp.n_actions)
                for transition in mdp.transitions[state][action]
            )
            if has_path:
                reachable.add(state)
                changed = True
    if len(reachable) != mdp.n_states:
        raise ValueError("gamma=1 value iteration requires every state to have a path to a terminal state")
    if any(
        transition.reward > 0.0
        for state in range(mdp.n_states)
        for action in range(mdp.n_actions)
        for transition in mdp.transitions[state][action]
    ):
        raise ValueError("gamma=1 value iteration requires non-positive rewards in this course")
    if initial_values is not None and np.any(_validate_values(mdp, initial_values) != 0.0):
        raise ValueError("gamma=1 value iteration must start from zero values in this course")


def bellman_expectation_backup(
    mdp: TabularMDP,
    state: int,
    values: Sequence[float] | FloatArray,
    policy_row: Sequence[float] | FloatArray,
    gamma: float,
) -> float:
    """对单个状态执行 Bellman expectation backup。"""
    values_array = _validate_values(mdp, values)
    state = mdp._validate_index(state, mdp.n_states, "state")
    row = np.asarray(policy_row, dtype=np.float64)
    if row.shape != (mdp.n_actions,):
        raise ValueError(f"policy_row must have shape ({mdp.n_actions},)")
    if (
        not np.all(np.isfinite(row))
        or np.any(row < 0)
        or not np.isclose(row.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("policy_row must be a probability distribution")
    gamma = TabularMDP._validate_gamma(gamma)
    return float(
        sum(
            row[action] * mdp.expected_action_return(state, action, values_array, gamma)
            for action in range(mdp.n_actions)
        )
    )


def iterative_policy_evaluation(
    mdp: TabularMDP,
    policy: Sequence[Sequence[float]] | FloatArray,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
    initial_values: Sequence[float] | FloatArray | None = None,
) -> EvaluationResult:
    """同步迭代求解固定策略的 Bellman expectation 方程。"""
    gamma, theta, max_iterations = _validate_solver_args(gamma, theta, max_iterations)
    policy_array = _validate_policy(mdp, policy)
    if gamma == 1.0 and not _policy_terminates_almost_surely(mdp, policy_array):
        raise ValueError("gamma=1 policy evaluation requires a proper episodic policy")
    values = (
        np.zeros(mdp.n_states, dtype=np.float64)
        if initial_values is None
        else _validate_values(mdp, initial_values).copy()
    )
    delta = float("inf")
    for iteration in range(1, max_iterations + 1):
        updated = np.empty_like(values)
        for state in range(mdp.n_states):
            updated[state] = bellman_expectation_backup(
                mdp, state, values, policy_array[state], gamma
            )
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta < theta:
            return EvaluationResult(values, iteration, True, delta)
    return EvaluationResult(values, max_iterations, False, delta)


def action_value_table(
    mdp: TabularMDP,
    values: Sequence[float] | FloatArray,
    gamma: float,
) -> FloatArray:
    """由 ``V`` 计算所有 ``Q(s,a)``，返回形状 ``(S,A)``。"""
    values_array = _validate_values(mdp, values)
    gamma = TabularMDP._validate_gamma(gamma)
    result = np.empty((mdp.n_states, mdp.n_actions), dtype=np.float64)
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            result[state, action] = mdp.expected_action_return(
                state, action, values_array, gamma
            )
    return result


def greedy_policy_improvement(
    mdp: TabularMDP,
    values: Sequence[float] | FloatArray,
    gamma: float = 0.99,
) -> tuple[FloatArray, FloatArray]:
    """对 ``V`` 贪心改进；最大值并列时在所有最优动作上均匀分配概率。"""
    action_values = action_value_table(mdp, values, gamma)
    policy = np.zeros_like(action_values)
    for state in range(mdp.n_states):
        best = np.max(action_values[state])
        winners = np.flatnonzero(action_values[state] == best)
        policy[state, winners] = 1.0 / winners.size
    return policy, action_values


def policy_iteration(
    mdp: TabularMDP,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 1_000,
    evaluation_max_iterations: int = 20_000,
) -> DPResult:
    """交替进行完整策略评估与贪心改进。"""
    gamma, theta, max_iterations = _validate_solver_args(gamma, theta, max_iterations)
    if (
        isinstance(evaluation_max_iterations, bool)
        or not isinstance(evaluation_max_iterations, Integral)
        or evaluation_max_iterations < 1
    ):
        raise ValueError("evaluation_max_iterations must be a positive integer")
    policy = np.full(
        (mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions, dtype=np.float64
    )
    values = np.zeros(mdp.n_states, dtype=np.float64)
    last_delta = float("inf")
    for iteration in range(1, max_iterations + 1):
        evaluation = iterative_policy_evaluation(
            mdp,
            policy,
            gamma=gamma,
            theta=theta,
            max_iterations=int(evaluation_max_iterations),
            initial_values=values,
        )
        values = evaluation.values
        last_delta = evaluation.delta
        if not evaluation.converged:
            return DPResult(values, policy, iteration, False, last_delta)
        improved, _ = greedy_policy_improvement(mdp, values, gamma)
        if np.array_equal(improved, policy):
            return DPResult(values, improved, iteration, True, last_delta)
        if iteration == max_iterations:
            # values 属于当前已评估 policy；不要与尚未评估的 improved 策略混装。
            return DPResult(values, policy, iteration, False, last_delta)
        policy = improved
    return DPResult(values, policy, max_iterations, False, last_delta)


def value_iteration(
    mdp: TabularMDP,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
    initial_values: Sequence[float] | FloatArray | None = None,
) -> DPResult:
    """迭代 Bellman optimality backup，并从收敛 ``V`` 导出贪心策略。"""
    gamma, theta, max_iterations = _validate_solver_args(gamma, theta, max_iterations)
    if gamma == 1.0:
        _validate_undiscounted_control_problem(mdp, initial_values)
    values = (
        np.zeros(mdp.n_states, dtype=np.float64)
        if initial_values is None
        else _validate_values(mdp, initial_values).copy()
    )
    delta = float("inf")
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        updated = np.max(action_value_table(mdp, values, gamma), axis=1)
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta < theta:
            converged = True
            break
    policy, _ = greedy_policy_improvement(mdp, values, gamma)
    return DPResult(values, policy, iteration, converged, delta)
