# src/replay_buffer.py

import numpy as np
import torch


class ReplayBuffer:

    def __init__(
        self,
        state_dim,
        action_dim,
        capacity=100_000,
    ):

        self.capacity = capacity

        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (capacity, action_dim),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (capacity, 1),
            dtype=np.float32,
        )

        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            (capacity, 1),
            dtype=np.float32,
        )

        self.pointer = 0
        self.size = 0

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        index = self.pointer

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        self.pointer = (
            self.pointer + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size,
        device,
    ):

        indices = np.random.randint(
            0,
            self.size,
            size=batch_size,
        )

        states = torch.as_tensor(
            self.states[indices],
            dtype=torch.float32,
            device=device,
        )

        actions = torch.as_tensor(
            self.actions[indices],
            dtype=torch.float32,
            device=device,
        )

        rewards = torch.as_tensor(
            self.rewards[indices],
            dtype=torch.float32,
            device=device,
        )

        next_states = torch.as_tensor(
            self.next_states[indices],
            dtype=torch.float32,
            device=device,
        )

        dones = torch.as_tensor(
            self.dones[indices],
            dtype=torch.float32,
            device=device,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __len__(self):

        return self.size