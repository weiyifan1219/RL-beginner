from typing import Callable, Dict, List, Tuple


State = Tuple[int, int]
Action = int
Transition = Tuple[State, Action, float]


def generate_episode(
    env,
    policy: Callable[[State], Action],
    max_steps: int = 1000,
) -> List[Transition]:
    """
    Generate one complete episode by interacting with the environment.

    Each transition is stored as:
        (state, action, reward)
    """
    episode = []

    state = env.reset()

    for _ in range(max_steps):
        action = policy(state)

        next_state, reward, done = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if done:
            return episode

    return RuntimeError(
        f"Episode did not terminate within {max_steps} steps."
    )


def first_visit_mc_prediction(
    env,
    policy: Callable[[State], Action],
    num_episodes: int,
    gamma: float = 1.0,
) -> Dict[State, float]:
    """
    First-Visit Monte Carlo prediction.

    Estimate V_pi(s) for a fixed policy pi using sampled episodes.
    """
    states = env.get_states()

    V = {state: 0.0 for state in states}
    counts = {state: 0 for state in states}

    for _ in range(num_episodes):

        # 1. Generate a complete episode
        episode = generate_episode(env, policy)

        # 2. Compute G_t for every timestep
        returns = [0.0] * len(episode)

        G = 0.0

        for t in reversed(range(len(episode))):
            _, _, reward = episode[t]

            G = reward + gamma * G
            returns[t] = G

        # 3. First-Visit MC update
        visited = set()

        for t, (state, _, _) in enumerate(episode):

            if state in visited:
                continue

            visited.add(state)

            G = returns[t]

            counts[state] += 1

            alpha = 1.0 / counts[state]

            V[state] += alpha * (G - V[state])

    return V