import torch
import torch.nn as nn
from torch.optim import Adam


class PPOAgent:
    """Clipped PPO with mini-batch updates and update diagnostics."""

    def __init__(
        self,
        actor,
        critic,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=4,
        minibatch_size=256,
    ):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state)
        return action.item(), log_prob.item(), value.item()

    def value(self, state):
        """Return V(state) without constructing a training graph."""
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.critic(state).item()

    def update(self, states, actions, old_log_probs, advantages, returns):
        """Perform PPO epochs over shuffled mini-batches.

        The returned ``approx_kl`` and ``clip_fraction`` are measured on the
        actual update batches, so experiments can test the PPO theory instead
        of inferring it from a reward curve alone.
        """
        states = states.detach()
        actions = actions.detach().long()
        old_log_probs = old_log_probs.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        n_samples = states.shape[0]
        if n_samples == 0:
            raise ValueError("PPO update received an empty rollout")
        batch_size = min(self.minibatch_size or n_samples, n_samples)

        totals = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "batches": 0,
        }

        for _ in range(self.update_epochs):
            indices = torch.randperm(n_samples, device=states.device)
            for start in range(0, n_samples, batch_size):
                batch = indices[start : start + batch_size]
                dist = self.actor(states[batch])
                new_log_probs = dist.log_prob(actions[batch])
                entropy = dist.entropy().mean()
                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * advantages[batch]
                actor_loss = -torch.minimum(surr1, surr2).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()

                values = self.critic(states[batch])
                critic_loss = self.value_coef * nn.functional.mse_loss(
                    values,
                    returns[batch],
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optimizer.step()

                with torch.no_grad():
                    totals["approx_kl"] += ((ratio - 1.0) - log_ratio).mean().item()
                    totals["clip_fraction"] += (
                        (torch.abs(ratio - 1.0) > self.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )
                    totals["entropy"] += entropy.item()
                totals["actor_loss"] += actor_loss.item()
                totals["critic_loss"] += critic_loss.item()
                totals["batches"] += 1

        count = float(totals.pop("batches"))
        return {name: value / count for name, value in totals.items()}
