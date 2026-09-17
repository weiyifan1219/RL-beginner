# src/td3.py

import copy

import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class TD3Agent:

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
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        device="cpu",
    ):

        self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.update_step = 0

        self.action_low = np.asarray(
            action_low,
            dtype=np.float32,
        )

        self.action_high = np.asarray(
            action_high,
            dtype=np.float32,
        )

        self.action_low_tensor = torch.as_tensor(
            action_low,
            dtype=torch.float32,
            device=self.device,
        )

        self.action_high_tensor = torch.as_tensor(
            action_high,
            dtype=torch.float32,
            device=self.device,
        )

        # =====================================================
        # Actor
        # =====================================================

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)

        self.actor_target = copy.deepcopy(
            self.actor
        ).to(self.device)


        # =====================================================
        # Twin Critics
        # =====================================================

        self.critic1 = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.critic2 = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.critic1_target = copy.deepcopy(
            self.critic1
        ).to(self.device)

        self.critic2_target = copy.deepcopy(
            self.critic2
        ).to(self.device)


        # =====================================================
        # Target networks do not receive gradients
        # =====================================================

        for network in [
            self.actor_target,
            self.critic1_target,
            self.critic2_target,
        ]:
            for param in network.parameters():
                param.requires_grad = False


        # =====================================================
        # Optimizers
        # =====================================================

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=critic_lr,
        )

        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=critic_lr,
        )


    # =========================================================
    # Select Action
    # =========================================================

    @torch.no_grad()
    def select_action(
        self,
        state,
        noise_std=0.0,
    ):

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

        action = np.clip(
            action,
            self.action_low,
            self.action_high,
        )

        return action.astype(
            np.float32
        )


    # =========================================================
    # Update
    # =========================================================

    def update(
        self,
        replay_buffer,
        batch_size,
    ):

        self.update_step += 1

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


        # =====================================================
        # 1. Target Policy Smoothing
        # =====================================================

        with torch.no_grad():

            # Target Actor
            next_actions = self.actor_target(
                next_states
            )

            # Small clipped Gaussian noise
            noise = torch.randn_like(
                next_actions
            ) * self.policy_noise

            noise = noise.clamp(
                -self.noise_clip,
                self.noise_clip,
            )

            next_actions = (
                next_actions + noise
            )

            # Keep action inside valid range
            next_actions = torch.max(
                torch.min(
                    next_actions,
                    self.action_high_tensor,
                ),
                self.action_low_tensor,
            )


            # =================================================
            # 2. Clipped Double Q-Learning
            # =================================================

            target_q1 = self.critic1_target(
                next_states,
                next_actions,
            )

            target_q2 = self.critic2_target(
                next_states,
                next_actions,
            )

            min_target_q = torch.min(
                target_q1,
                target_q2,
            )


            # TD Target
            target_q = (
                rewards
                + self.gamma
                * (1.0 - dones)
                * min_target_q
            )


        # =====================================================
        # 3. Update Critic 1
        # =====================================================

        current_q1 = self.critic1(
            states,
            actions,
        )

        critic1_loss = F.mse_loss(
            current_q1,
            target_q,
        )

        self.critic1_optimizer.zero_grad()

        critic1_loss.backward()

        self.critic1_optimizer.step()


        # =====================================================
        # 4. Update Critic 2
        # =====================================================

        current_q2 = self.critic2(
            states,
            actions,
        )

        critic2_loss = F.mse_loss(
            current_q2,
            target_q,
        )

        self.critic2_optimizer.zero_grad()

        critic2_loss.backward()

        self.critic2_optimizer.step()


        actor_loss_value = None


        # =====================================================
        # 5. Delayed Policy Update
        # =====================================================

        if (
            self.update_step
            % self.policy_delay
            == 0
        ):

            policy_actions = self.actor(
                states
            )

            actor_q = self.critic1(
                states,
                policy_actions,
            )

            actor_loss = -actor_q.mean()

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            self.actor_optimizer.step()

            actor_loss_value = (
                actor_loss.item()
            )


            # =================================================
            # 6. Soft Update
            #
            # TD3 中 Target Network 也跟随 delayed update
            # =================================================

            self._soft_update(
                self.actor,
                self.actor_target,
            )

            self._soft_update(
                self.critic1,
                self.critic1_target,
            )

            self._soft_update(
                self.critic2,
                self.critic2_target,
            )


        return {
            "actor_loss":
                actor_loss_value,

            "critic1_loss":
                critic1_loss.item(),

            "critic2_loss":
                critic2_loss.item(),

            "q1":
                current_q1.mean().item(),

            "q2":
                current_q2.mean().item(),

            "target_q":
                target_q.mean().item(),
        }


    # =========================================================
    # Soft Update
    # =========================================================

    def _soft_update(
        self,
        source_network,
        target_network,
    ):

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