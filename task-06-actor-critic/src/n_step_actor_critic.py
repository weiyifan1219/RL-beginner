import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .networks import Actor, Critic


class NStepActorCriticAgent:
    """
    N-step Actor-Critic

    n-step Target:

        G_t^(n)
        =
        r_t
        + gamma r_{t+1}
        + ...
        + gamma^(n-1) r_{t+n-1}
        + gamma^n V(s_{t+n})

    Advantage:

        A_t
        =
        G_t^(n) - V(s_t)

    Actor Loss:

        L_actor
        =
        -log pi(a_t | s_t) * A_t

    Critic Loss:

        L_critic
        =
        MSE(V(s_t), G_t^(n))
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

    def select_action(self, state):
        """
        根据当前策略 pi(a|s) 采样动作。
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
        rewards,
        bootstrap_state,
        terminated,
    ):
        """
        使用 n-step trajectory 更新一个 (s_t, a_t)。

        Parameters
        ----------
        state:
            s_t

        action:
            a_t

        rewards:
            [r_t, r_{t+1}, ..., r_{t+n-1}]

        bootstrap_state:
            s_{t+n}

        terminated:
            最后一步是否真正进入 terminal state
        """

        # =====================================================
        # 1. Tensor
        # =====================================================

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action = torch.tensor(
            [action],
            dtype=torch.long,
            device=self.device,
        )

        bootstrap_state = torch.as_tensor(
            bootstrap_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # =====================================================
        # 2. 计算 n-step Target
        # =====================================================

        with torch.no_grad():

            # 如果真正 terminal：
            #
            # V(terminal) = 0
            if terminated:

                R = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=self.device,
                )

            else:

                # 没有 terminal：
                #
                # 从 s_{t+n} bootstrap
                R = self.critic(
                    bootstrap_state
                ).squeeze()

            # -------------------------------------------------
            # 从后往前递推：
            #
            # R_t = r_t + gamma R_{t+1}
            # -------------------------------------------------

            for reward in reversed(rewards):

                reward = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=self.device,
                )

                R = (
                    reward
                    + self.gamma * R
                )

            n_step_target = (
                R.unsqueeze(0)
            )

        # =====================================================
        # 3. Critic
        # =====================================================

        value = self.critic(
            state
        )

        advantage = (
            n_step_target - value
        )

        critic_loss = F.mse_loss(
            value,
            n_step_target,
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # =====================================================
        # 4. Actor
        # =====================================================

        logits = self.actor(
            state
        )

        dist = Categorical(
            logits=logits
        )

        log_prob = dist.log_prob(
            action
        )

        actor_loss = -(
            log_prob
            * advantage.detach()
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "advantage": advantage.mean().item(),
            "value": value.mean().item(),
            "n_step_target": n_step_target.mean().item(),
        }