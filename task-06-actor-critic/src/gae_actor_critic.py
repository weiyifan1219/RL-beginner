import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .networks import Actor, Critic


class GAEActorCriticAgent:
    """
    Actor-Critic with Generalized Advantage Estimation (GAE)

    TD Error:

        delta_t =
            r_t
            + gamma * V(s_{t+1})
            - V(s_t)

    GAE:

        A_t =
            delta_t
            + gamma * lambda * A_{t+1}

    Actor Loss:

        L_actor =
            -log pi(a_t | s_t) * A_t

    Critic Target:

        R_t =
            A_t + V(s_t)

    Critic Loss:

        L_critic =
            MSE(V(s_t), R_t)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.device = torch.device(
            device
        )

        # =====================================================
        # Actor / Critic
        # =====================================================

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic = Critic(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # =====================================================
        # Optimizers
        # =====================================================

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

    def select_action(
        self,
        state,
    ):
        """
        根据当前策略采样动作。
        """

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        logits = self.actor(
            state
        )

        dist = Categorical(
            logits=logits
        )

        action = dist.sample()

        return action.item()

    def update(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminateds,
    ):
        """
        使用一整段 rollout 计算 GAE，
        然后更新 Actor 和 Critic。

        Parameters
        ----------
        states:
            [s_0, s_1, ..., s_{T-1}]

        actions:
            [a_0, a_1, ..., a_{T-1}]

        rewards:
            [r_0, r_1, ..., r_{T-1}]

        next_states:
            [s_1, s_2, ..., s_T]

        terminateds:
            是否真正进入 terminal state
        """

        # =====================================================
        # 1. Tensor
        # =====================================================

        states = torch.as_tensor(
            np.asarray(
                states,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.asarray(
                next_states,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        terminateds = torch.as_tensor(
            terminateds,
            dtype=torch.float32,
            device=self.device,
        )

        # =====================================================
        # 2. Critic Values
        # =====================================================

        values = self.critic(
            states
        )

        with torch.no_grad():

            next_values = self.critic(
                next_states
            )

            # terminal 后不能继续 bootstrap
            masks = (
                1.0 - terminateds
            )

            # =================================================
            # 3. One-step TD Error
            #
            # delta_t =
            # r_t + gamma V(s_{t+1}) - V(s_t)
            # =================================================

            deltas = (
                rewards
                + self.gamma
                * next_values
                * masks
                - values.detach()
            )

            # =================================================
            # 4. GAE
            #
            # A_t =
            # delta_t
            # + gamma lambda A_{t+1}
            # =================================================

            advantages = torch.zeros_like(
                rewards
            )

            gae = torch.tensor(
                0.0,
                device=self.device,
            )

            for t in reversed(
                range(len(rewards))
            ):

                gae = (
                    deltas[t]
                    + self.gamma
                    * self.gae_lambda
                    * masks[t]
                    * gae
                )

                advantages[t] = gae

            # =================================================
            # 5. Critic Target
            #
            # R_t = A_t + V(s_t)
            # =================================================

            returns = (
                advantages
                + values.detach()
            )

        # =====================================================
        # 6. Critic Update
        # =====================================================

        critic_loss = F.mse_loss(
            values,
            returns,
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # =====================================================
        # 7. Actor Update
        # =====================================================

        logits = self.actor(
            states
        )

        dist = Categorical(
            logits=logits
        )

        log_probs = dist.log_prob(
            actions
        )

        actor_loss = -(
            log_probs
            * advantages.detach()
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),

            "advantage_mean": (
                advantages.mean().item()
            ),

            "advantage_std": (
                advantages.std().item()
            ),

            "value_mean": (
                values.mean().item()
            ),

            "return_mean": (
                returns.mean().item()
            ),
        }