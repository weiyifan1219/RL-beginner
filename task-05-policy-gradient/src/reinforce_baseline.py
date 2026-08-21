import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical


# ============================================================
# 1. Policy Network
# ============================================================

class PolicyNetwork(nn.Module):
    """
    Policy Network

    输入：
        state s

    输出：
        pi(a | s)
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

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        """
        输入 state，输出 action probabilities。
        """

        logits = self.net(state)

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        return probs

    def sample_action(self, state):
        """
        根据当前策略概率分布采样动作。

        Returns
        -------
        action:
            采样得到的动作

        log_prob:
            log pi(a | s)
        """

        probs = self.forward(state)

        dist = Categorical(
            probs=probs
        )

        action = dist.sample()

        log_prob = dist.log_prob(
            action
        )

        return action, log_prob


# ============================================================
# 2. Value Network
# ============================================================

class ValueNetwork(nn.Module):
    """
    State Value Network

    输入：
        state s

    输出：
        V(s)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        """
        返回 V(s)
        """

        value = self.net(state)

        return value.squeeze(-1)


# ============================================================
# 3. Monte Carlo Return
# ============================================================

def compute_returns(
    rewards,
    gamma=0.99,
):
    """
    计算一个完整 Episode 中每一步的 discounted return。

    G_t = r_t + gamma * G_{t+1}

    Parameters
    ----------
    rewards:
        一个 Episode 的 reward list

    gamma:
        discount factor

    Returns
    -------
    returns:
        Tensor [G_0, G_1, ..., G_T]
    """

    returns = []

    G = 0.0

    # 从 Episode 最后一步向前递推
    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.append(G)

    # 前面是倒序计算，所以翻回来
    returns.reverse()

    returns = torch.tensor(
        returns,
        dtype=torch.float32,
    )

    return returns


# ============================================================
# 4. REINFORCE + Baseline Update
# ============================================================

def update_policy_with_baseline(
    policy_optimizer,
    value_optimizer,
    value_network,
    states,
    log_probs,
    rewards,
    gamma=0.99,
):
    """
    使用一个完整 Episode 更新 Policy Network 和 Value Network。

    Policy:
        A_t = G_t - V(s_t)

        L_policy
            = - mean(
                A_t * log pi(a_t | s_t)
              )

    Value:
        L_value
            = mean(
                (V(s_t) - G_t)^2
              )
    """

    # --------------------------------------------------------
    # Step 1
    # 计算 Monte Carlo Returns
    # --------------------------------------------------------

    returns = compute_returns(
        rewards=rewards,
        gamma=gamma,
    )

    # --------------------------------------------------------
    # Step 2
    # Episode states
    # --------------------------------------------------------

    states = torch.stack(
        states
    )

    # --------------------------------------------------------
    # Step 3
    # Value Network 预测 V(s_t)
    # --------------------------------------------------------

    values = value_network(
        states
    )

    # --------------------------------------------------------
    # Step 4
    # Advantage
    #
    # A_t = G_t - V(s_t)
    #
    # G_t：
    #   当前 trajectory 得到的真实 Monte Carlo Return
    #
    # V(s_t)：
    #   Value Network 对正常水平的预测
    # --------------------------------------------------------

    advantages = (
        returns
        - values.detach()
    )

    # --------------------------------------------------------
    # Step 5
    # Policy Loss
    #
    # L = - A_t log pi(a_t | s_t)
    # --------------------------------------------------------

    log_probs = torch.stack(
        log_probs
    )

    policy_loss = -(
        log_probs
        * advantages
    ).mean()

    # --------------------------------------------------------
    # Step 6
    # Value Loss
    #
    # 用真实采样 Return G_t
    # 监督 V(s_t)
    # --------------------------------------------------------

    value_loss = (
        values
        - returns
    ).pow(2).mean()

    # --------------------------------------------------------
    # Step 7
    # Update Policy
    # --------------------------------------------------------

    policy_optimizer.zero_grad()

    policy_loss.backward()

    policy_optimizer.step()

    # --------------------------------------------------------
    # Step 8
    # Update Value Network
    # --------------------------------------------------------

    value_optimizer.zero_grad()

    value_loss.backward()

    value_optimizer.step()

    return {
        "policy_loss": policy_loss.item(),

        "value_loss": value_loss.item(),

        "returns": returns.detach(),

        "values": values.detach(),

        "advantages": advantages.detach(),
    }


# ============================================================
# 5. Training
# ============================================================

def train_reinforce_baseline(
    env,
    num_episodes=1000,
    gamma=0.99,
    policy_lr=1e-3,
    value_lr=1e-3,
    hidden_dim=128,
    seed=42,
    print_interval=50,
):
    """
    完整训练 REINFORCE with Baseline。

    Parameters
    ----------
    env:
        Gymnasium environment

    num_episodes:
        训练 Episode 数量

    gamma:
        discount factor

    policy_lr:
        Policy Network learning rate

    value_lr:
        Value Network learning rate

    hidden_dim:
        hidden dimension

    seed:
        random seed

    print_interval:
        每隔多少 Episode 打印一次
    """

    # ========================================================
    # Random Seed
    # ========================================================

    np.random.seed(seed)

    torch.manual_seed(seed)

    # ========================================================
    # Environment dimensions
    # ========================================================

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n

    # ========================================================
    # Networks
    # ========================================================

    policy_network = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )

    value_network = ValueNetwork(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
    )

    # ========================================================
    # Optimizers
    # ========================================================

    policy_optimizer = optim.Adam(
        policy_network.parameters(),
        lr=policy_lr,
    )

    value_optimizer = optim.Adam(
        value_network.parameters(),
        lr=value_lr,
    )

    # ========================================================
    # Training history
    # ========================================================

    history = {
        "episode_returns": [],
        "episode_lengths": [],
        "policy_losses": [],
        "value_losses": [],
        "mean_advantages": [],
    }

    # ========================================================
    # Training Loop
    # ========================================================

    for episode in range(
        num_episodes
    ):

        # ----------------------------------------------------
        # Reset Environment
        # ----------------------------------------------------

        if episode == 0:

            state, info = env.reset(
                seed=seed
            )

        else:

            state, info = env.reset()

        # ----------------------------------------------------
        # Episode buffer
        # ----------------------------------------------------

        states = []

        log_probs = []

        rewards = []

        terminated = False

        truncated = False

        # ----------------------------------------------------
        # Collect one full Episode
        # ----------------------------------------------------

        while not (
            terminated
            or truncated
        ):

            # numpy -> tensor
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
            )

            # Policy Network
            action, log_prob = (
                policy_network.sample_action(
                    state_tensor
                )
            )

            # Environment Step
            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action.item()
            )

            # Save trajectory data
            states.append(
                state_tensor
            )

            log_probs.append(
                log_prob
            )

            rewards.append(
                reward
            )

            # Move to next state
            state = next_state

        # ====================================================
        # Episode finished
        #
        # 现在已经获得完整 rewards：
        #
        # r0, r1, ..., rT
        #
        # 因此现在才能计算：
        #
        # G0, G1, ..., GT
        # ====================================================

        update_info = (
            update_policy_with_baseline(
                policy_optimizer=policy_optimizer,

                value_optimizer=value_optimizer,

                value_network=value_network,

                states=states,

                log_probs=log_probs,

                rewards=rewards,

                gamma=gamma,
            )
        )

        # ====================================================
        # Statistics
        # ====================================================

        episode_return = sum(
            rewards
        )

        episode_length = len(
            rewards
        )

        mean_advantage = (
            update_info[
                "advantages"
            ]
            .mean()
            .item()
        )

        history[
            "episode_returns"
        ].append(
            episode_return
        )

        history[
            "episode_lengths"
        ].append(
            episode_length
        )

        history[
            "policy_losses"
        ].append(
            update_info[
                "policy_loss"
            ]
        )

        history[
            "value_losses"
        ].append(
            update_info[
                "value_loss"
            ]
        )

        history[
            "mean_advantages"
        ].append(
            mean_advantage
        )

        # ====================================================
        # Print
        # ====================================================

        if (
            episode + 1
        ) % print_interval == 0:

            avg_return = np.mean(
                history[
                    "episode_returns"
                ][
                    -print_interval:
                ]
            )

            avg_value_loss = np.mean(
                history[
                    "value_losses"
                ][
                    -print_interval:
                ]
            )

            print(
                f"Episode {episode + 1:4d} | "
                f"Return {episode_return:6.1f} | "
                f"Avg Return {avg_return:7.2f} | "
                f"Policy Loss "
                f"{update_info['policy_loss']:8.4f} | "
                f"Value Loss "
                f"{avg_value_loss:8.4f}"
            )

    return (
        policy_network,
        value_network,
        history,
    )


# ============================================================
# 6. Evaluation
# ============================================================

def evaluate_policy(
    env,
    policy_network,
    num_episodes=20,
    deterministic=True,
):
    """
    Evaluation Policy。

    deterministic=True:
        使用 argmax，不再随机采样。

    训练时：
        sample action

    测试时：
        通常直接选择概率最大的 action
    """

    returns = []

    for _ in range(
        num_episodes
    ):

        state, info = env.reset()

        terminated = False

        truncated = False

        episode_return = 0

        while not (
            terminated
            or truncated
        ):

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
            )

            with torch.no_grad():

                probs = policy_network(
                    state_tensor
                )

            if deterministic:

                action = torch.argmax(
                    probs
                ).item()

            else:

                dist = Categorical(
                    probs=probs
                )

                action = (
                    dist.sample()
                    .item()
                )

            (
                state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            episode_return += reward

        returns.append(
            episode_return
        )

    return {
        "mean_return": np.mean(
            returns
        ),

        "std_return": np.std(
            returns
        ),

        "returns": returns,
    }