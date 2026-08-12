from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.agents import EpsilonGreedyAgent, ThompsonSamplingAgent
from src.bandits import BernoulliBandit
from src.experiment import run_episode, run_experiment


TASK_DIR = Path(__file__).resolve().parents[1]


class AlwaysBestAgent:
    def __init__(self) -> None:
        self.n_arms = 2

    def select_action(self) -> int:
        return 1

    def update(self, action: int, reward: float) -> None:
        del action, reward


def test_run_episode_records_reward_optimal_action_and_expected_regret() -> None:
    result = run_episode(BernoulliBandit([0.0, 1.0], seed=0), AlwaysBestAgent(), n_steps=8)

    np.testing.assert_array_equal(result.actions, np.ones(8, dtype=np.int64))
    np.testing.assert_array_equal(result.rewards, np.ones(8))
    np.testing.assert_array_equal(result.optimal_actions, np.ones(8, dtype=bool))
    np.testing.assert_array_equal(result.instantaneous_regret, np.zeros(8))
    np.testing.assert_array_equal(result.cumulative_regret, np.zeros(8))


def test_aggregated_experiment_is_reproducible_and_has_documented_shapes() -> None:
    env_factory = lambda seed: BernoulliBandit([0.2, 0.8], seed=seed)
    agent_factory = lambda seed: EpsilonGreedyAgent(2, epsilon=0.1, seed=seed)

    first = run_experiment(env_factory, agent_factory, n_runs=12, n_steps=40, seed=9)
    second = run_experiment(env_factory, agent_factory, n_runs=12, n_steps=40, seed=9)

    assert first.n_runs == 12
    assert first.n_steps == 40
    for field in (
        "mean_reward",
        "reward_standard_error",
        "optimal_action_rate",
        "mean_instantaneous_regret",
        "mean_cumulative_regret",
    ):
        left = getattr(first, field)
        right = getattr(second, field)
        assert left.shape == (40,)
        np.testing.assert_array_equal(left, right)
    assert np.all(np.diff(first.mean_cumulative_regret) >= -1e-12)


def test_thompson_sampling_reaches_high_optimal_action_rate() -> None:
    result = run_experiment(
        lambda seed: BernoulliBandit([0.1, 0.3, 0.9], seed=seed),
        lambda seed: ThompsonSamplingAgent(3, seed=seed),
        n_runs=40,
        n_steps=300,
        seed=123,
    )

    assert result.optimal_action_rate[-50:].mean() > 0.9
    assert result.mean_cumulative_regret[-1] < 30.0


def test_quick_cli_writes_machine_readable_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "quick-run"
    completed = subprocess.run(
        [
            sys.executable,
            str(TASK_DIR / "run_experiment.py"),
            "--quick",
            "--output-dir",
            str(output_dir),
            "--no-plot",
        ],
        cwd=TASK_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_runs"] > 0
    assert summary["n_steps"] > 0
    assert set(summary["agents"]) == {"epsilon-greedy", "ucb", "thompson-sampling"}
    assert (output_dir / "curves.npz").is_file()
