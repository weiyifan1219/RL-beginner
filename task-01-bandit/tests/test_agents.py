from __future__ import annotations

import numpy as np
import pytest

from src.agents import (
    EpsilonGreedyAgent,
    ThompsonSamplingAgent,
    UCBAgent,
    incremental_update,
)
from src.bandits import BernoulliBandit


def test_incremental_update_supports_sample_average_and_constant_step() -> None:
    assert incremental_update(estimate=2.0, reward=5.0, count=3) == pytest.approx(3.0)
    assert incremental_update(estimate=2.0, reward=5.0, count=3, step_size=0.2) == pytest.approx(2.6)


@pytest.mark.parametrize("count", [0, -1])
def test_incremental_update_requires_positive_count(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        incremental_update(estimate=0.0, reward=1.0, count=count)


def test_epsilon_greedy_exploits_unique_best_estimate() -> None:
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.0, seed=11)
    agent.estimates[:] = [0.0, 3.0, 1.0]

    assert [agent.select_action() for _ in range(10)] == [1] * 10


def test_epsilon_greedy_updates_sample_average() -> None:
    agent = EpsilonGreedyAgent(n_arms=2, epsilon=0.0, seed=0)

    agent.update(1, 2.0)
    agent.update(1, 4.0)

    np.testing.assert_array_equal(agent.counts, [0, 2])
    np.testing.assert_allclose(agent.estimates, [0.0, 3.0])


def test_ucb_selects_every_untried_arm_before_reusing_one() -> None:
    agent = UCBAgent(n_arms=4, c=2.0, seed=13)
    selected = []

    for _ in range(4):
        action = agent.select_action()
        selected.append(action)
        agent.update(action, 0.0)

    assert set(selected) == {0, 1, 2, 3}


def test_thompson_sampling_updates_beta_posterior() -> None:
    agent = ThompsonSamplingAgent(n_arms=2, alpha_prior=1.0, beta_prior=1.0, seed=0)

    agent.update(0, 1.0)
    agent.update(0, 0.0)

    np.testing.assert_allclose(agent.alphas, [2.0, 1.0])
    np.testing.assert_allclose(agent.betas, [2.0, 1.0])
    np.testing.assert_array_equal(agent.counts, [2, 0])
    with pytest.raises(ValueError, match="Bernoulli"):
        agent.update(1, 0.5)


def test_thompson_sampling_learns_an_obviously_better_bernoulli_arm() -> None:
    env = BernoulliBandit([0.05, 0.95], seed=17)
    agent = ThompsonSamplingAgent(n_arms=2, seed=23)
    actions = []

    for _ in range(1_000):
        action = agent.select_action()
        actions.append(action)
        agent.update(action, env.pull(action))

    assert np.mean(np.asarray(actions[-300:]) == 1) > 0.9
