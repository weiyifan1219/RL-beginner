import random
from typing import Dict, Tuple


State = Tuple[int, int]
Action = int


def epsilon_greedy(
    Q: Dict[State, Dict[Action, float]],
    state: State,
    actions,
    epsilon: float,
) -> Action:
    """
    Select an action using epsilon-greedy policy.
    """

    if random.random() < epsilon:
        return random.choice(actions)

    q_values = Q[state]

    max_q = max(q_values.values())

    best_actions = [
        action
        for action in actions
        if q_values[action] == max_q
    ]

    return random.choice(best_actions)


def q_learning(
    env,
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    max_steps: int = 1000,
):
    """
    Tabular Q-Learning control.

    Update:
        Q(s, a) <- Q(s, a) + alpha * [
            r + gamma * max_a' Q(s', a') - Q(s, a)
        ]
    """

    states = env.get_states()
    actions = list(env.get_actions())

    Q = {
        state: {
            action: 0.0
            for action in actions
        }
        for state in states
    }

    episode_returns = []

    for _ in range(num_episodes):

        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):

            # Behavior policy: epsilon-greedy
            action = epsilon_greedy(
                Q,
                state,
                actions,
                epsilon,
            )

            next_state, reward, done = env.step(action)

            total_reward += reward

            if done:
                td_target = reward

            else:
                # Target policy: greedy
                best_next_q = max(
                    Q[next_state].values()
                )

                td_target = (
                    reward
                    + gamma * best_next_q
                )

            td_error = (
                td_target
                - Q[state][action]
            )

            Q[state][action] += (
                alpha * td_error
            )

            state = next_state

            if done:
                break

        episode_returns.append(total_reward)

    return Q, episode_returns