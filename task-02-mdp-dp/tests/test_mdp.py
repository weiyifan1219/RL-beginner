from __future__ import annotations

import numpy as np
import pytest

from src.mdp import TabularMDP, Transition


def tiny_mdp() -> TabularMDP:
    return TabularMDP(
        [
            [
                [(1.0, 0, 1.0, False)],
                [(0.25, 0, 0.0, False), (0.75, 1, 2.0, True)],
            ],
            [
                [(1.0, 1, 0.0, True)],
                [(1.0, 1, 0.0, True)],
            ],
        ]
    )


def test_transition_is_a_frozen_explicit_record() -> None:
    transition = Transition(0.5, 1, -2.0, True)

    assert transition.probability == 0.5
    assert transition.next_state == 1
    assert transition.reward == -2.0
    assert transition.terminated is True


def test_expected_action_return_uses_probabilities_and_terminal_mask() -> None:
    mdp = tiny_mdp()
    values = np.array([4.0, 100.0])

    assert mdp.expected_action_return(0, 0, values, gamma=0.5) == pytest.approx(3.0)
    # 0.25 * (0 + .5*4) + .75 * (2 + 0*100) = 2.0
    assert mdp.expected_action_return(0, 1, values, gamma=0.5) == pytest.approx(2.0)


def test_transition_table_is_normalized_and_read_only() -> None:
    mdp = tiny_mdp()

    assert mdp.n_states == 2
    assert mdp.n_actions == 2
    assert isinstance(mdp.transitions[0][0][0], Transition)
    with pytest.raises(TypeError):
        mdp.transitions[0][0][0] = Transition(1.0, 0, 0.0, False)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ([], "state"),
        ([[[]]], "non-empty"),
        ([[[(0.8, 0, 0.0, False)]]], "sum"),
        ([[[(-0.1, 0, 0.0, False), (1.1, 0, 0.0, False)]]], "probability"),
        ([[[(1.0, 1, 0.0, False)]]], "next_state"),
        ([[[(1.0, 0, np.nan, False)]]], "reward"),
        ([[[(1.0, 0, 0.0, 1)]]], "terminated"),
        (
            [
                [[(1.0, 0, 0.0, False)]],
                [[(1.0, 1, 0.0, False)], [(1.0, 1, 0.0, False)]],
            ],
            "same number of actions",
        ),
    ],
)
def test_invalid_transition_tables_are_rejected(table, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        TabularMDP(table)


def test_expected_return_validates_shapes_indices_and_gamma() -> None:
    mdp = tiny_mdp()

    with pytest.raises(ValueError, match="shape"):
        mdp.expected_action_return(0, 0, np.zeros(3), gamma=0.9)
    with pytest.raises(IndexError, match="state"):
        mdp.expected_action_return(2, 0, np.zeros(2), gamma=0.9)
    with pytest.raises(IndexError, match="action"):
        mdp.expected_action_return(0, 2, np.zeros(2), gamma=0.9)
    with pytest.raises(ValueError, match="gamma"):
        mdp.expected_action_return(0, 0, np.zeros(2), gamma=1.1)
