# src/sac.py

import copy

import numpy as np
import torch
import torch.nn.functional as F

from networks import (
    GaussianActor,
    Critic,
)


class SACAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        actor_lr=3e-4,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        init_alpha=0.2,
        autotune=True,
        device="cpu",
    ):

        self.device = torch.device(
            device
        )

        self.gamma = gamma
        self.tau = tau

        self.autotune = autotune


        # ====================================================
        # Actor
        # ====================================================

        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(
            self.device
        )


        # ====================================================
        # Twin Critics
        # ====================================================

        self.critic1 = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(
            self.device
        )

        self.critic2 = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(
            self.device
        )


        # ====================================================
        # Target Critics
        #
        # SAC 通常没有 Target Actor
        # ====================================================

        self.critic1_target = (
            copy.deepcopy(
                self.critic1
            )
            .to(self.device)
        )

        self.critic2_target = (
            copy.deepcopy(
                self.critic2
            )
            .to(self.device)
        )


        for network in [
            self.critic1_target,
            self.critic2_target,
        ]:

            for param in (
                network.parameters()
            ):
                param.requires_grad = False


        # ====================================================
        # Optimizers
        # ====================================================

        self.actor_optimizer = (
            torch.optim.Adam(
                self.actor.parameters(),
                lr=actor_lr,
            )
        )

        self.critic1_optimizer = (
            torch.optim.Adam(
                self.critic1.parameters(),
                lr=critic_lr,
            )
        )

        self.critic2_optimizer = (
            torch.optim.Adam(
                self.critic2.parameters(),
                lr=critic_lr,
            )
        )


        # ====================================================
        # Temperature alpha
        # ====================================================

        if self.autotune:

            # 常见默认：
            #
            # target_entropy = -action_dim
            #
            self.target_entropy = float(
                -action_dim
            )

            self.log_alpha = torch.tensor(
                np.log(init_alpha),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

            self.alpha_optimizer = (
                torch.optim.Adam(
                    [self.log_alpha],
                    lr=alpha_lr,
                )
            )

        else:

            self.log_alpha = None

            self.alpha = float(
                init_alpha
            )


    # ========================================================
    # alpha
    # ========================================================

    @property
    def current_alpha(
        self,
    ):

        if self.autotune:

            return (
                self.log_alpha.exp()
            )

        return torch.tensor(
            self.alpha,
            dtype=torch.float32,
            device=self.device,
        )


    # ========================================================
    # Action Selection
    # ========================================================

    @torch.no_grad()
    def select_action(
        self,
        state,
        deterministic=False,
    ):

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        (
            sampled_action,
            _,
            mean_action,
        ) = self.actor.sample(
            state
        )


        if deterministic:

            action = mean_action

        else:

            action = sampled_action


        return (
            action
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )


    # ========================================================
    # Update
    # ========================================================

    def update(
        self,
        replay_buffer,
        batch_size,
    ):

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


        alpha = self.current_alpha


        # ====================================================
        # 1. Build Soft Bellman Target
        # ====================================================

        with torch.no_grad():

            (
                next_actions,
                next_log_probs,
                _,
            ) = self.actor.sample(
                next_states
            )


            target_q1 = (
                self.critic1_target(
                    next_states,
                    next_actions,
                )
            )

            target_q2 = (
                self.critic2_target(
                    next_states,
                    next_actions,
                )
            )


            min_target_q = torch.min(
                target_q1,
                target_q2,
            )


            # ================================================
            # Soft Q target
            #
            # Q - alpha * log π
            # ================================================

            next_soft_value = (
                min_target_q
                - alpha
                * next_log_probs
            )


            target_q = (
                rewards
                + self.gamma
                * (1.0 - dones)
                * next_soft_value
            )


        # ====================================================
        # 2. Update Critic 1
        # ====================================================

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


        # ====================================================
        # 3. Update Critic 2
        # ====================================================

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


        # ====================================================
        # 4. Update Actor
        # ====================================================

        (
            policy_actions,
            log_probs,
            _,
        ) = self.actor.sample(
            states
        )


        q1_policy = self.critic1(
            states,
            policy_actions,
        )

        q2_policy = self.critic2(
            states,
            policy_actions,
        )


        min_policy_q = torch.min(
            q1_policy,
            q2_policy,
        )


        # ================================================
        # Actor:
        #
        # minimize:
        #
        # alpha log π(a|s) - Q(s,a)
        # ================================================

        actor_loss = (
            alpha.detach()
            * log_probs
            - min_policy_q
        ).mean()


        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()


        # ====================================================
        # 5. Automatic Entropy Tuning
        # ====================================================

        alpha_loss_value = None


        if self.autotune:

            alpha_loss = (
                -self.log_alpha.exp()
                * (
                    log_probs.detach()
                    + self.target_entropy
                )
            ).mean()


            self.alpha_optimizer.zero_grad()

            alpha_loss.backward()

            self.alpha_optimizer.step()


            alpha_loss_value = (
                alpha_loss.item()
            )


        # ====================================================
        # 6. Soft Update Target Critics
        # ====================================================

        self._soft_update(
            self.critic1,
            self.critic1_target,
        )

        self._soft_update(
            self.critic2,
            self.critic2_target,
        )


        # ====================================================
        # Logging
        # ====================================================

        return {

            "actor_loss":
                actor_loss.item(),

            "critic1_loss":
                critic1_loss.item(),

            "critic2_loss":
                critic2_loss.item(),

            "alpha":
                self.current_alpha
                .detach()
                .item(),

            "alpha_loss":
                alpha_loss_value,

            "log_prob":
                log_probs
                .mean()
                .item(),

            "q1":
                current_q1
                .mean()
                .item(),

            "q2":
                current_q2
                .mean()
                .item(),

            "target_q":
                target_q
                .mean()
                .item(),
        }


    # ========================================================
    # Soft Update
    # ========================================================

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