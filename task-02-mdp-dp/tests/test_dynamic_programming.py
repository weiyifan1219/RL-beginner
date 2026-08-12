from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.dynamic_programming import (
    bellman_expectation_backup,
    greedy_policy_improvement,
    iterative_policy_evaluation,
    policy_iteration,
    value_iteration,
)
from src.gridworld import GridWorld
from src.mdp import TabularMDP


TASK_DIR = Path(__file__).resolve().parents[1]


def continuing_one_state_mdp() -> TabularMDP:
    return TabularMDP([[[(1.0, 0, 1.0, False)]]])


def choose_between_terminal_rewards() -> TabularMDP:
    return TabularMDP(
        [
            [[(1.0, 1, 0.0, True)], [(1.0, 1, 2.0, True)]],
            [[(1.0, 1, 0.0, True)], [(1.0, 1, 0.0, True)]],
        ]
    )


def test_bellman_expectation_backup_matches_hand_calculation() -> None:
    mdp = choose_between_terminal_rewards()
    policy_row = np.array([0.25, 0.75])

    updated = bellman_expectation_backup(mdp, 0, np.array([99.0, 100.0]), policy_row, gamma=0.9)

    assert updated == pytest.approx(1.5)


def test_bellman_backup_rejects_a_policy_row_that_does_not_sum_to_one() -> None:
    mdp = choose_between_terminal_rewards()

    with pytest.raises(ValueError, match="probability distribution"):
        bellman_expectation_backup(
            mdp,
            0,
            np.zeros(2),
            np.array([0.5000025, 0.5000025]),
            gamma=0.9,
        )


def test_policy_evaluation_solves_geometric_return() -> None:
    result = iterative_policy_evaluation(
        continuing_one_state_mdp(),
        policy=np.ones((1, 1)),
        gamma=0.5,
        theta=1e-12,
        max_iterations=1_000,
    )

    assert result.converged
    assert result.values[0] == pytest.approx(2.0, abs=1e-10)
    assert result.iterations > 1
    assert result.delta < 1e-12


def test_gamma_one_rejects_an_improper_nonterminating_policy() -> None:
    with pytest.raises(ValueError, match="proper episodic"):
        iterative_policy_evaluation(
            continuing_one_state_mdp(),
            policy=np.ones((1, 1)),
            gamma=1.0,
            initial_values=np.array([42.0]),
        )


def test_gamma_one_value_iteration_requires_a_path_to_termination() -> None:
    with pytest.raises(ValueError, match="terminal state"):
        value_iteration(continuing_one_state_mdp(), gamma=1.0)


def test_policy_validation_rejects_bad_shapes_and_probabilities() -> None:
    mdp = choose_between_terminal_rewards()

    with pytest.raises(ValueError, match="shape"):
        iterative_policy_evaluation(mdp, np.ones((2, 1)), gamma=0.9)
    with pytest.raises(ValueError, match="non-negative"):
        iterative_policy_evaluation(mdp, np.array([[1.2, -0.2], [0.5, 0.5]]), gamma=0.9)
    with pytest.raises(ValueError, match="sum"):
        iterative_policy_evaluation(mdp, np.full((2, 2), 0.4), gamma=0.9)


def test_greedy_improvement_selects_best_action_and_splits_exact_ties() -> None:
    mdp = choose_between_terminal_rewards()
    policy, action_values = greedy_policy_improvement(mdp, np.zeros(2), gamma=0.9)

    np.testing.assert_allclose(action_values[0], [0.0, 2.0])
    np.testing.assert_allclose(policy[0], [0.0, 1.0])
    np.testing.assert_allclose(policy[1], [0.5, 0.5])


def test_greedy_improvement_does_not_include_a_nearly_best_action() -> None:
    gap = 5e-13
    mdp = TabularMDP(
        [
            [[(1.0, 1, 1.0, True)], [(1.0, 1, 1.0 - gap, True)]],
            [[(1.0, 1, 0.0, True)], [(1.0, 1, 0.0, True)]],
        ]
    )

    policy, action_values = greedy_policy_improvement(mdp, np.zeros(2), gamma=1.0)

    assert action_values[0, 0] > action_values[0, 1]
    np.testing.assert_array_equal(policy[0], [1.0, 0.0])


def test_gridworld_uniform_policy_matches_sutton_golden_values() -> None:
    env = GridWorld(rows=4, cols=4)
    result = iterative_policy_evaluation(
        env,
        env.uniform_random_policy(),
        gamma=1.0,
        theta=1e-11,
        max_iterations=20_000,
    )
    expected = np.array(
        [
            [0.0, -14.0, -20.0, -22.0],
            [-14.0, -18.0, -20.0, -20.0],
            [-20.0, -20.0, -18.0, -14.0],
            [-22.0, -20.0, -14.0, 0.0],
        ]
    )

    assert result.converged
    np.testing.assert_allclose(result.values.reshape(4, 4), expected, atol=1e-7)


def test_policy_and_value_iteration_agree_on_gridworld_optimum() -> None:
    env = GridWorld(rows=4, cols=4)
    policy_result = policy_iteration(env, gamma=1.0, theta=1e-11, max_iterations=100)
    value_result = value_iteration(env, gamma=1.0, theta=1e-11, max_iterations=10_000)
    expected = np.array(
        [
            [0.0, -1.0, -2.0, -3.0],
            [-1.0, -2.0, -3.0, -2.0],
            [-2.0, -3.0, -2.0, -1.0],
            [-3.0, -2.0, -1.0, 0.0],
        ]
    )

    assert policy_result.converged and value_result.converged
    np.testing.assert_allclose(policy_result.values, value_result.values, atol=1e-8)
    np.testing.assert_allclose(value_result.values.reshape(4, 4), expected, atol=1e-8)
    np.testing.assert_allclose(policy_result.policy, value_result.policy, atol=1e-12)
    np.testing.assert_allclose(value_result.policy.sum(axis=1), 1.0)


def test_nonconvergence_is_reported_not_silently_accepted() -> None:
    result = iterative_policy_evaluation(
        continuing_one_state_mdp(), np.ones((1, 1)), gamma=0.99, theta=1e-15, max_iterations=2
    )

    assert not result.converged
    assert result.iterations == 2
    assert result.delta > 1e-15


def test_exhausted_policy_iteration_returns_values_for_the_returned_policy() -> None:
    mdp = choose_between_terminal_rewards()
    result = policy_iteration(mdp, gamma=0.0, theta=1e-12, max_iterations=1)

    assert not result.converged
    np.testing.assert_allclose(result.policy, 0.5)
    # The returned uniform policy has value 1 at state 0; the improved policy would have value 2.
    assert result.values[0] == pytest.approx(1.0)


def test_quick_cli_writes_documented_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(TASK_DIR / "run_experiment.py"),
            "--quick",
            "--no-plot",
            "--output-dir",
            str(output_dir),
        ],
        cwd=TASK_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected_files = {"resolved_config.yaml", "summary.json", "values.npy", "policy.npy"}
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["algorithm"] == "value_iteration"
    assert summary["converged"] is True
    assert np.load(output_dir / "values.npy").shape == (16,)
    assert np.load(output_dir / "policy.npy").shape == (16, 4)
