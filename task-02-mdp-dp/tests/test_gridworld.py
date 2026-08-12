from __future__ import annotations

import numpy as np
import pytest

from src.gridworld import ACTION_NAMES, GridWorld


def test_gridworld_state_coordinate_round_trip_and_action_names() -> None:
    env = GridWorld(rows=4, cols=4)

    assert ACTION_NAMES == ("up", "right", "down", "left")
    for state in range(env.n_states):
        assert env.coord_to_state(env.state_to_coord(state)) == state


def test_gridworld_boundaries_and_terminal_entry() -> None:
    env = GridWorld(rows=4, cols=4)

    # state 1 at (0, 1): up hits the boundary; left enters terminal state 0.
    up = env.transitions[1][0][0]
    left = env.transitions[1][3][0]
    assert (up.next_state, up.reward, up.terminated) == (1, -1.0, False)
    assert (left.next_state, left.reward, left.terminated) == (0, -1.0, True)


def test_terminal_states_are_absorbing_with_zero_future_reward() -> None:
    env = GridWorld(rows=4, cols=4)

    for terminal in env.terminal_states:
        for action in range(env.n_actions):
            transition = env.transitions[terminal][action][0]
            assert transition.next_state == terminal
            assert transition.reward == 0.0
            assert transition.terminated is True


def test_gridworld_validates_size_and_coordinates() -> None:
    with pytest.raises(ValueError, match="rows"):
        GridWorld(rows=1, cols=4)
    env = GridWorld(rows=4, cols=4)
    with pytest.raises(IndexError, match="state"):
        env.state_to_coord(16)
    with pytest.raises(IndexError, match="coordinate"):
        env.coord_to_state((4, 0))


def test_uniform_random_policy_has_correct_shape_and_rows() -> None:
    env = GridWorld(rows=4, cols=4)
    policy = env.uniform_random_policy()

    assert policy.shape == (16, 4)
    np.testing.assert_allclose(policy.sum(axis=1), 1.0)
    np.testing.assert_allclose(policy, 0.25)
