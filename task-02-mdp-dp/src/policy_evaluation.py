# src/policy_evaluation.py

from typing import Dict, Callable

from src.gridworld import GridWorld, State


ValueTable = Dict[State, float]
Policy = Callable[[State], Dict[int, float]]


def create_uniform_random_policy(env: GridWorld) -> Policy:
    """
    Create a uniform random policy.

    Every action has the same probability:
        pi(a | s) = 1 / |A|
    """

    action_prob = 1.0 / len(env.ACTIONS)

    def policy(state: State):
        return {
            action: action_prob
            for action in env.ACTIONS
        }

    return policy


def policy_evaluation(
    env,
    policy,
    gamma=0.9,
    theta=1e-6,
    return_history=False,
):
    states = env.get_states()

    values = {
        state: 0.0
        for state in states
    }

    history = [values.copy()]

    while True:
        delta = 0.0
        new_values = values.copy()

        for state in states:

            if env.is_terminal(state):
                new_values[state] = 0.0
                continue

            state_value = 0.0
            action_probs = policy(state)

            for action, action_prob in action_probs.items():

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

                state_value += (
                    action_prob * q_value
                )

            new_values[state] = state_value

            delta = max(
                delta,
                abs(
                    new_values[state]
                    - values[state]
                ),
            )

        values = new_values
        history.append(values.copy())

        if delta < theta:
            break

    if return_history:
        return values, history

    return values