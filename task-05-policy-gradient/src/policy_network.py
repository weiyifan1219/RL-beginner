import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    Policy Network for discrete action spaces.

    Input:
        state

    Output:
        action probability distribution pi(a | s)
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
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        """
        Return action probabilities.
        """
        logits = self.net(state)

        probs = torch.softmax(logits, dim=-1)

        return probs

    def sample_action(self, state):
        """
        Sample an action according to pi(a | s).

        Returns:
            action
            log_prob
        """

        probs = self.forward(state)

        dist = Categorical(probs=probs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob