import torch


def compute_gae(
    rewards,
    values,
    dones,
    last_value,
    gamma=0.99,
    gae_lambda=0.95,
):
    """
    Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : Tensor [T]
        R_t

    values : Tensor [T]
        V(s_t)

    dones : Tensor [T]
        真实 episode 是否结束

    last_value : Tensor or float
        rollout 最后状态 s_T 的 V(s_T)

        如果 rollout 恰好在 terminal 结束，应为 0；
        如果只是 rollout_steps 截断，则需要 bootstrap。

    gamma : float

    gae_lambda : float

    Returns
    -------
    advantages : Tensor [T]

    returns : Tensor [T]
        Critic target:
        advantage + V(s_t)
    """

    T = len(rewards)

    advantages = torch.zeros_like(
        rewards,
        dtype=torch.float32,
    )

    gae = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=rewards.device,
    )

    last_value = torch.as_tensor(
        last_value,
        dtype=torch.float32,
        device=rewards.device,
    )


    for t in reversed(range(T)):

        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        # terminal后不bootstrap
        non_terminal = 1.0 - dones[t]

        delta = (
            rewards[t]
            + gamma * next_value * non_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * non_terminal
            * gae
        )

        advantages[t] = gae


    returns = (
        advantages
        + values
    )

    return (
        advantages,
        returns,
    )