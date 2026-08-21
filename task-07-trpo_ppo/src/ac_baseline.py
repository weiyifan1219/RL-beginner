import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam


class ActorCriticAgent:
    """The one-step TD Actor-Critic baseline from Task 06.

    PPO is compared with this agent under the same number of environment
    transitions.  The target bootstraps across time-limit truncation and only
    stops at a true MDP terminal state.
    """

    def __init__(
        self,
        actor,
        critic,
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.99,
    ):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).sample()
        return action.item()

    def update(self, state, action, reward, next_state, terminated):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)

        value = self.critic(state)
        with torch.no_grad():
            next_value = self.critic(next_state)
            target = reward if terminated else reward + self.gamma * next_value

        td_error = target - value
        critic_loss = F.mse_loss(value, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        dist = self.actor(state)
        actor_loss = -(dist.log_prob(action) * td_error.detach()).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": td_error.item(),
        }
