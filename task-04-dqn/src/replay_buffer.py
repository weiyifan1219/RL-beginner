from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """
    Store transitions:
        (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        transition = (
            state,
            action,
            reward,
            next_state,
            done,
        )

        self.buffer.append(transition)

    def sample(self, batch_size: int):
        transitions = random.sample(
            self.buffer,
            batch_size,
        )

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __len__(self):
        return len(self.buffer)