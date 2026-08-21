import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .networks import Actor, Critic


class ActorCriticAgent:
    """
    One-step Actor-Critic

    TD Target:
        r_t + gamma * V(s_{t+1})

    TD Error:
        delta_t
        =
        r_t + gamma * V(s_{t+1}) - V(s_t)

    Actor:
        L_actor
        =
        -log π(a_t|s_t) * delta_t

    Critic:
        L_critic
        =
        MSE(V(s_t), TD Target)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.99,
        device="cpu",
    ):
        self.gamma = gamma
        self.device = torch.device(device)

        # =========================
        # Actor
        # =========================

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # =========================
        # Critic
        # =========================

        self.critic = Critic(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # =========================
        # Optimizers
        # =========================

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

    def select_action(self, state):
        """
        根据 π(a|s) 采样动作
        """

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        logits = self.actor(state)

        dist = Categorical(
            logits=logits
        )

        action = dist.sample()

        return action.item()

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):
        """
        使用一个 transition：

        (s_t, a_t, r_t, s_{t+1})

        进行一次更新。
        """

        # =========================
        # Tensor
        # =========================

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        next_state = torch.as_tensor(
            next_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action = torch.tensor(
            [action],
            dtype=torch.long,
            device=self.device,
        )

        reward = torch.tensor(
            [reward],
            dtype=torch.float32,
            device=self.device,
        )

        # =========================
        # V(s_t)
        # =========================

        value = self.critic(state)

        # =========================
        # TD Target
        # =========================

        with torch.no_grad():

            next_value = self.critic(
                next_state
            )

            if terminated:
                td_target = reward

            else:
                td_target = (
                    reward
                    + self.gamma * next_value
                )

        # =========================
        # TD Error
        # =========================

        td_error = (
            td_target - value
        )

        # =========================
        # Critic Update
        #
        # 注意：
        # 恢复最初版本的 MSE
        # =========================

        critic_loss = F.mse_loss(
            value,
            td_target,
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # =========================
        # Actor Update
        # =========================

        logits = self.actor(state)

        dist = Categorical(
            logits=logits
        )

        log_prob = dist.log_prob(
            action
        )

        actor_loss = -(
            log_prob
            * td_error.detach()
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": td_error.mean().item(),
            "value": value.mean().item(),
            "td_target": td_target.mean().item(),
        }