from typing import Callable, Dict, Tuple


State = Tuple[int, int]
Action = int


def td0_prediction(
    env,
    policy: Callable[[State], Action],
    num_episodes: int,
    alpha: float = 0.05,
    gamma: float = 1.0,
    max_steps: int = 1000,
) -> Dict[State, float]:

    states = env.get_states()

    V = {state: 0.0 for state in states}

    for _ in range(num_episodes):

        state = env.reset()

        for _ in range(max_steps):

            action = policy(state)

            next_state, reward, done = env.step(action)

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]

            td_error = td_target - V[state]

            V[state] += alpha * td_error

            state = next_state

            if done:
                break

    return V