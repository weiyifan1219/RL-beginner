import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Actor:
        s -> π(a|s)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """
    Critic:
        s -> V(s)
    """

    def __init__(
        self,
        state_dim,
        hidden_dim=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)