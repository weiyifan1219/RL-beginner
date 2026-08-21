# src/value_iteration.py

from typing import Dict, Tuple

from src.gridworld import GridWorld, State


ValueTable = Dict[State, float]
Policy = Dict[State, int]


def value_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-6,
) -> Tuple[ValueTable, Policy]:

    states = env.get_states()

    # V_0(s) = 0
    values = {
        state: 0.0
        for state in states
    }

    while True:

        delta = 0.0
        new_values = values.copy()

        for state in states:

            if env.is_terminal(state):
                new_values[state] = 0.0
                continue

            action_values = []

            for action in env.ACTIONS:

                next_state = env.get_next_state(
                    state,
                    action,
                )

                reward = env.get_reward(
                    state,
                    action,
                    next_state,
                )

                q_value = (
                    reward
                    + gamma * values[next_state]
                )

                action_values.append(q_value)

            new_values[state] = max(action_values)

            delta = max(
                delta,
                abs(
                    new_values[state]
                    - values[state]
                ),
            )

        values = new_values

        if delta < theta:
            break

    # --------------------------------
    # Extract optimal policy
    # --------------------------------

    policy = {}

    for state in states:

        if env.is_terminal(state):
            continue

        best_action = None
        best_q_value = float("-inf")

        for action in env.ACTIONS:

            next_state = env.get_next_state(
                state,
                action,
            )

            reward = env.get_reward(
                state,
                action,
                next_state,
            )

            q_value = (
                reward
                + gamma * values[next_state]
            )

            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        policy[state] = best_action

    return values, policy