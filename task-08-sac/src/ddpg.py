# src/ddpg.py

import copy

import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class DDPGAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        device="cpu",
    ):
        """
        DDPG Agent.

        Parameters
        ----------
        state_dim : int
            状态维度

        action_dim : int
            动作维度

        action_low : np.ndarray
            动作空间下界

        action_high : np.ndarray
            动作空间上界

        actor_lr : float
            Actor learning rate

        critic_lr : float
            Critic learning rate

        gamma : float
            Discount factor

        tau : float
            Soft update coefficient

        device : str
            cpu / cuda
        """

        self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau

        self.action_low = np.asarray(
            action_low,
            dtype=np.float32,
        )

        self.action_high = np.asarray(
            action_high,
            dtype=np.float32,
        )

        # ======================================================
        # 1. Online Actor
        # ======================================================

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)

        # ======================================================
        # 2. Target Actor
        # ======================================================

        self.actor_target = copy.deepcopy(
            self.actor
        ).to(self.device)

        # ======================================================
        # 3. Online Critic
        # ======================================================

        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)

        # ======================================================
        # 4. Target Critic
        # ======================================================

        self.critic_target = copy.deepcopy(
            self.critic
        ).to(self.device)

        # Target networks 不通过 optimizer 训练
        for param in self.actor_target.parameters():
            param.requires_grad = False

        for param in self.critic_target.parameters():
            param.requires_grad = False

        # ======================================================
        # Optimizers
        # ======================================================

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

    # ==========================================================
    # Action Selection
    # ==========================================================

    @torch.no_grad()
    def select_action(
        self,
        state,
        noise_std=0.0,
    ):
        """
        根据当前 Actor 选择动作。

        训练:
            action = actor(state) + exploration noise

        测试:
            noise_std = 0
        """

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action = self.actor(
            state
        ).cpu().numpy()[0]

        # Exploration noise
        if noise_std > 0:

            noise = np.random.normal(
                loc=0.0,
                scale=noise_std,
                size=action.shape,
            )

            action = action + noise

        # 防止噪声导致动作越界
        action = np.clip(
            action,
            self.action_low,
            self.action_high,
        )

        return action.astype(
            np.float32
        )

    # ==========================================================
    # Training
    # ==========================================================

    def update(
        self,
        replay_buffer,
        batch_size,
    ):
        """
        One DDPG gradient update.
        """

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = replay_buffer.sample(
            batch_size=batch_size,
            device=self.device,
        )

        # ======================================================
        # 1. Compute TD Target
        # ======================================================

        with torch.no_grad():

            # a' = μ_target(s')
            next_actions = self.actor_target(
                next_states
            )

            # Q_target(s', a')
            next_q_values = self.critic_target(
                next_states,
                next_actions,
            )

            # y = r + γ(1-d)Q_target(s', μ_target(s'))
            target_q_values = (
                rewards
                + self.gamma
                * (1.0 - dones)
                * next_q_values
            )

        # ======================================================
        # 2. Update Critic
        # ======================================================

        current_q_values = self.critic(
            states,
            actions,
        )

        critic_loss = F.mse_loss(
            current_q_values,
            target_q_values,
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # ======================================================
        # 3. Update Actor
        # ======================================================

        policy_actions = self.actor(
            states
        )

        actor_q_values = self.critic(
            states,
            policy_actions,
        )

        # Maximize Q
        # PyTorch optimizer minimizes loss,
        # so use negative Q.
        actor_loss = -actor_q_values.mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # ======================================================
        # 4. Soft Update Target Networks
        # ======================================================

        self._soft_update(
            self.actor,
            self.actor_target,
        )

        self._soft_update(
            self.critic,
            self.critic_target,
        )

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_value": current_q_values.mean().item(),
            "target_q": target_q_values.mean().item(),
        }

    # ==========================================================
    # Soft Update
    # ==========================================================

    def _soft_update(
        self,
        source_network,
        target_network,
    ):
        """
        θ_target <- τ θ + (1 - τ) θ_target
        """

        with torch.no_grad():

            for source_param, target_param in zip(
                source_network.parameters(),
                target_network.parameters(),
            ):

                target_param.data.mul_(
                    1.0 - self.tau
                )

                target_param.data.add_(
                    self.tau
                    * source_param.data
                )