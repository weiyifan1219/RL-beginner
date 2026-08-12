from __future__ import annotations

import numpy as np
import pytest

from src.bandits import BernoulliBandit, GaussianBandit, NonStationaryGaussianBandit


def test_bernoulli_bandit_has_exact_deterministic_arms() -> None:
    env = BernoulliBandit([0.0, 1.0], seed=7)

    assert [env.pull(0) for _ in range(5)] == [0.0] * 5
    assert [env.pull(1) for _ in range(5)] == [1.0] * 5
    np.testing.assert_allclose(env.expected_rewards, [0.0, 1.0])
    assert env.n_arms == 2
    assert env.optimal_arm == 1
    assert env.optimal_mean == 1.0
    assert env.regret(0) == pytest.approx(1.0)
    assert env.regret(1) == pytest.approx(0.0)


def test_bandit_reset_replays_the_same_random_sequence() -> None:
    env = BernoulliBandit([0.25, 0.75], seed=19)
    first = [env.pull(i % 2) for i in range(30)]

    env.reset(seed=19)
    second = [env.pull(i % 2) for i in range(30)]

    assert first == second


def test_gaussian_bandits_are_reproducible_and_validate_actions() -> None:
    left = GaussianBandit([-1.0, 2.0], std=0.5, seed=3)
    right = GaussianBandit([-1.0, 2.0], std=0.5, seed=3)

    np.testing.assert_allclose(
        [left.pull(1) for _ in range(20)],
        [right.pull(1) for _ in range(20)],
    )
    with pytest.raises(IndexError, match="action"):
        left.pull(2)


def test_zero_variance_gaussian_is_a_deterministic_teaching_environment() -> None:
    env = GaussianBandit([-0.5, 1.5], std=0.0, seed=99)

    assert [env.pull(0), env.pull(1), env.pull(1)] == [-0.5, 1.5, 1.5]


@pytest.mark.parametrize(
    ("constructor", "values"),
    [
        (BernoulliBandit, []),
        (BernoulliBandit, [-0.1, 0.5]),
        (BernoulliBandit, [0.5, 1.1]),
        (GaussianBandit, []),
        (GaussianBandit, [0.0, np.nan]),
    ],
)
def test_bandit_rejects_invalid_reward_parameters(constructor, values) -> None:
    with pytest.raises(ValueError):
        constructor(values)


def test_nonstationary_bandit_drifts_and_reset_restores_initial_means() -> None:
    env = NonStationaryGaussianBandit(
        [0.0, 0.0, 0.0], std=0.1, random_walk_std=0.2, seed=5
    )
    initial = env.expected_rewards.copy()

    env.pull(0)
    assert not np.array_equal(env.expected_rewards, initial)

    env.reset(seed=5)
    np.testing.assert_array_equal(env.expected_rewards, initial)
