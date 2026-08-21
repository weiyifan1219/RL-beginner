from typing import Dict

from src.gridworld import GridWorld, State
from src.policy_evaluation import policy_evaluation


DeterministicPolicy = Dict[State, int]


def deterministic_policy_to_callable(
    env: GridWorld,
    policy: DeterministicPolicy,
):
    """
    Convert deterministic action mapping into
    pi(a|s) probability form.
    """

    def policy_fn(state: State):

        best_action = policy[state]

        return {
            action: 1.0 if action == best_action else 0.0
            for action in env.ACTIONS
        }

    return policy_fn


def policy_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-6,
):

    states = env.get_states()

    # 初始 deterministic policy
    # 所有状态暂时选择 UP
    policy = {
        state: env.UP
        for state in states
        if not env.is_terminal(state)
    }

    while True:

        # -------------------------
        # 1. Policy Evaluation
        # -------------------------

        policy_fn = deterministic_policy_to_callable(
            env,
            policy,
        )

        values = policy_evaluation(
            env,
            policy_fn,
            gamma=gamma,
            theta=theta,
        )

        # -------------------------
        # 2. Policy Improvement
        # -------------------------

        policy_stable = True

        for state in states:

            if env.is_terminal(state):
                continue

            old_action = policy[state]

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

            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return policy, values