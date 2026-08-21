import torch


class RolloutBuffer:
    """
    PPO Rollout Buffer

    保存一次 policy rollout 的数据。
    数据来自当前 policy，因此是 on-policy 数据。

    ``dones`` stores the true MDP terminal flag used by GAE.  A Gymnasium
    time-limit truncation should reset the environment but should not be
    stored as terminal, otherwise the value bootstrap is discarded.
    """

    def __init__(self):

        self.states = []

        self.actions = []

        self.rewards = []

        self.dones = []

        self.values = []

        self.log_probs = []


    def add(
        self,
        state,
        action,
        reward,
        done,
        value,
        log_prob,
    ):
        """
        保存一个 transition

        state:
            当前状态 s_t

        action:
            当前执行动作 a_t

        reward:
            环境返回 R_t

        done:
            是否进入真正的 terminal state；不要把 ``truncated`` 传进来

        value:
            Critic估计 V(s_t)

        log_prob:
            old policy下
            log π_old(a_t|s_t)
        """

        self.states.append(state)

        self.actions.append(action)

        self.rewards.append(reward)

        self.dones.append(bool(done))

        self.values.append(value)

        self.log_probs.append(log_prob)


    def clear(self):

        self.states.clear()

        self.actions.clear()

        self.rewards.clear()

        self.dones.clear()

        self.values.clear()

        self.log_probs.clear()


    def __len__(self):

        return len(self.states)
