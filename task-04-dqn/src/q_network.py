import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Neural network used to approximate the action-value function Q(s, a).

    Input:
        state: [batch_size, state_dim]

    Output:
        q_values: [batch_size, action_dim]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
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