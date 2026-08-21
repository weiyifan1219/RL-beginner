import numpy as np
import torch


def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted Monte Carlo returns.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Args:
        rewards: list of rewards from one episode
        gamma: discount factor

    Returns:
        returns: torch.Tensor, shape [T]
    """

    returns = []

    G = 0.0

    # 从 episode 最后一步向前计算
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)

    # 因为上面是倒着计算的，所以需要翻转回来
    returns.reverse()

    return torch.tensor(
        returns,
        dtype=torch.float32,
    )


def update_policy(
    optimizer,
    log_probs,
    rewards,
    gamma=0.99,
    normalize_returns=True,
):
    """
    Update policy using REINFORCE.

    Policy loss:
        L = - sum_t G_t * log pi(a_t | s_t)

    Args:
        optimizer:
            PyTorch optimizer

        log_probs:
            list of log pi(a_t | s_t)

        rewards:
            rewards collected from one episode

        gamma:
            discount factor

        normalize_returns:
            whether to normalize returns

    Returns:
        loss_value
        returns
    """

    returns = compute_returns(
        rewards=rewards,
        gamma=gamma,
    )

    if normalize_returns and len(returns) > 1:
        returns = (
            returns - returns.mean()
        ) / (
            returns.std() + 1e-8
        )

    log_probs = torch.stack(log_probs)

    loss = -(log_probs * returns).sum()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item(), returns.detach()