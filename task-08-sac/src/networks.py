# src/networks.py

import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Deterministic policy network.

    Input:
        state: [batch_size, state_dim]

    Output:
        action: [batch_size, action_dim]
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        hidden_dim=256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # 保存动作上下界
        action_low = torch.as_tensor(
            action_low,
            dtype=torch.float32,
        )

        action_high = torch.as_tensor(
            action_high,
            dtype=torch.float32,
        )

        # [-1, 1] -> [action_low, action_high]
        self.register_buffer(
            "action_scale",
            (action_high - action_low) / 2.0,
        )

        self.register_buffer(
            "action_bias",
            (action_high + action_low) / 2.0,
        )

    def forward(self, state):

        action = self.net(state)

        action = (
            action * self.action_scale
            + self.action_bias
        )

        return action


class Critic(nn.Module):
    """
    Q-function network.

    Input:
        state:  [batch_size, state_dim]
        action: [batch_size, action_dim]

    Output:
        Q(s, a): [batch_size, 1]
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                state_dim + action_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, state, action):

        x = torch.cat(
            [state, action],
            dim=-1,
        )

        q_value = self.net(x)

        return q_value 
    
# ============================================================
# SAC Gaussian Actor
# ============================================================

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    """
    Stochastic policy for SAC.

    Input:
        state

    Output:
        Gaussian distribution parameters:
            mean
            log_std
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        hidden_dim=256,
    ):
        super().__init__()

        # ====================================================
        # Shared feature network
        # ====================================================

        self.backbone = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        # ====================================================
        # Gaussian parameters
        # ====================================================

        self.mean_layer = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.log_std_layer = nn.Linear(
            hidden_dim,
            action_dim,
        )

        # ====================================================
        # Action scaling
        # ====================================================

        action_low = torch.as_tensor(
            action_low,
            dtype=torch.float32,
        )

        action_high = torch.as_tensor(
            action_high,
            dtype=torch.float32,
        )

        self.register_buffer(
            "action_scale",
            (action_high - action_low) / 2.0,
        )

        self.register_buffer(
            "action_bias",
            (action_high + action_low) / 2.0,
        )


    def forward(
        self,
        state,
    ):

        features = self.backbone(
            state
        )

        mean = self.mean_layer(
            features
        )

        log_std = self.log_std_layer(
            features
        )

        # 限制 std 范围，避免数值爆炸
        log_std = torch.tanh(
            log_std
        )

        log_std = (
            LOG_STD_MIN
            + 0.5
            * (LOG_STD_MAX - LOG_STD_MIN)
            * (log_std + 1.0)
        )

        return (
            mean,
            log_std,
        )


    def sample(
        self,
        state,
    ):
        """
        Reparameterized action sampling.

        Returns
        -------
        action:
            sampled stochastic action

        log_prob:
            log π(a|s)

        mean_action:
            deterministic action for evaluation
        """

        mean, log_std = self(
            state
        )

        std = log_std.exp()

        # Gaussian distribution
        distribution = (
            torch.distributions.Normal(
                mean,
                std,
            )
        )

        # ====================================================
        # Reparameterization Trick
        #
        # z = mean + std * epsilon
        # ====================================================

        z = distribution.rsample()

        # [-inf, inf] -> [-1, 1]
        tanh_action = torch.tanh(
            z
        )

        # [-1, 1] -> environment action range
        action = (
            tanh_action
            * self.action_scale
            + self.action_bias
        )

        # ====================================================
        # log π(a|s)
        # ====================================================

        log_prob = distribution.log_prob(
            z
        )

        # tanh transformation correction
        log_prob -= torch.log(
            self.action_scale
            * (
                1.0
                - tanh_action.pow(2)
            )
            + 1e-6
        )

        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True,
        )

        # ====================================================
        # Deterministic mean action
        #
        # Evaluation 使用
        # ====================================================

        mean_action = (
            torch.tanh(mean)
            * self.action_scale
            + self.action_bias
        )

        return (
            action,
            log_prob,
            mean_action,
        )