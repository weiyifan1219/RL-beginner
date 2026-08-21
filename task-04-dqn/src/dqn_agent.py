import numpy as np
import torch
import torch.nn.functional as F

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        buffer_capacity=10000,
        target_update_freq=100,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=lr,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity
        )

        self.update_count = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state)

        return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # prediction
        q_values = self.q_network(states)

        current_q = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        # TD target
        with torch.no_grad():
            next_q_values = self.target_network(
                next_states
            )

            max_next_q = next_q_values.max(
                dim=1
            ).values

            targets = (
                rewards
                + self.gamma
                * (1.0 - dones)
                * max_next_q
            )

        loss = F.mse_loss(
            current_q,
            targets,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        if (
            self.update_count
            % self.target_update_freq
            == 0
        ):
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

        return loss.item()