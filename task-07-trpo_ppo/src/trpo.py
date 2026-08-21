import torch
import torch.nn.functional as F
from torch.optim import Adam


class TRPOAgent:

    def __init__(
        self,
        actor,
        critic,
        critic_lr=1e-3,
        max_kl=0.01,
        damping=0.1,
        cg_iters=10,
        backtrack_iters=10,
        backtrack_coeff=0.8,
        critic_updates=10,
    ):

        self.actor = actor
        self.critic = critic

        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        # Trust Region大小
        self.max_kl = max_kl

        # Hessian数值稳定项
        self.damping = damping

        # Conjugate Gradient迭代次数
        self.cg_iters = cg_iters

        # Backtracking Line Search
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff

        # Critic普通梯度下降次数
        self.critic_updates = critic_updates


    # =========================================================
    # Rollout
    # =========================================================

    def select_action(
        self,
        state,
    ):

        state = torch.tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)


        with torch.no_grad():

            dist = self.actor(
                state
            )

            action = dist.sample()

            log_prob = dist.log_prob(
                action
            )

            value = self.critic(
                state
            )


        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )


    def get_value(
        self,
        state,
    ):

        state = torch.tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)


        with torch.no_grad():

            value = self.critic(
                state
            )


        return value.item()


    # =========================================================
    # Parameter utilities
    # =========================================================

    def _flat_params(self):

        return torch.cat(
            [
                p.data.view(-1)
                for p in self.actor.parameters()
            ]
        )


    def _set_flat_params(
        self,
        flat_params,
    ):

        index = 0

        for param in self.actor.parameters():

            numel = param.numel()

            param.data.copy_(
                flat_params[
                    index:index + numel
                ].view_as(param)
            )

            index += numel


    def _flat_grad(
        self,
        output,
        create_graph=False,
    ):

        grads = torch.autograd.grad(
            output,
            self.actor.parameters(),
            create_graph=create_graph,
        )

        return torch.cat(
            [
                grad.contiguous().view(-1)
                for grad in grads
            ]
        )


    # =========================================================
    # KL Divergence
    # =========================================================

    def _mean_kl(
        self,
        states,
        old_probs,
    ):
        """
        KL(
            π_old || π_new
        )

        old_probs固定，
        new policy来自当前actor。
        """

        new_dist = self.actor(
            states
        )

        new_probs = (
            new_dist.probs
            .clamp_min(1e-8)
        )

        old_probs_safe = (
            old_probs
            .clamp_min(1e-8)
        )


        kl = (
            old_probs_safe
            *
            (
                torch.log(
                    old_probs_safe
                )
                -
                torch.log(
                    new_probs
                )
            )
        ).sum(
            dim=-1
        )


        return kl.mean()


    # =========================================================
    # Hessian Vector Product
    # =========================================================

    def _hessian_vector_product(
        self,
        states,
        old_probs,
        vector,
    ):
        """
        不显式构造Hessian/Fisher矩阵 H。

        直接计算:

            H @ vector

        并加入 damping:

            (H + damping * I) @ vector
        """

        kl = self._mean_kl(
            states,
            old_probs,
        )


        kl_grad = self._flat_grad(
            kl,
            create_graph=True,
        )


        grad_vector_product = (
            kl_grad * vector
        ).sum()


        hvp = torch.autograd.grad(
            grad_vector_product,
            self.actor.parameters(),
        )


        flat_hvp = torch.cat(
            [
                grad.contiguous().view(-1)
                for grad in hvp
            ]
        )


        return (
            flat_hvp
            +
            self.damping
            *
            vector
        )


    # =========================================================
    # Conjugate Gradient
    # =========================================================

    def _conjugate_gradient(
        self,
        states,
        old_probs,
        b,
    ):
        """
        求解:

            Hx = b

        不计算:

            H^{-1}

        而用CG近似得到:

            x ≈ H^{-1} b
        """

        x = torch.zeros_like(
            b
        )

        r = b.clone()

        p = b.clone()

        r_dot_r = torch.dot(
            r,
            r
        )


        for _ in range(
            self.cg_iters
        ):

            Hp = (
                self._hessian_vector_product(
                    states,
                    old_probs,
                    p,
                )
            )


            alpha = (
                r_dot_r
                /
                (
                    torch.dot(
                        p,
                        Hp
                    )
                    + 1e-8
                )
            )


            x = (
                x
                +
                alpha * p
            )


            r = (
                r
                -
                alpha * Hp
            )


            new_r_dot_r = torch.dot(
                r,
                r
            )


            if new_r_dot_r < 1e-10:
                break


            beta = (
                new_r_dot_r
                /
                (
                    r_dot_r
                    + 1e-8
                )
            )


            p = (
                r
                +
                beta * p
            )


            r_dot_r = (
                new_r_dot_r
            )


        return x


    # =========================================================
    # Surrogate Objective
    # =========================================================

    def _surrogate_objective(
        self,
        states,
        actions,
        old_log_probs,
        advantages,
    ):

        dist = self.actor(
            states
        )


        new_log_probs = (
            dist.log_prob(
                actions
            )
        )


        ratio = torch.exp(
            new_log_probs
            -
            old_log_probs
        )


        objective = (
            ratio
            *
            advantages
        ).mean()


        return objective


    # =========================================================
    # TRPO Actor Update
    # =========================================================

    def _update_actor(
        self,
        states,
        actions,
        old_log_probs,
        advantages,
    ):

        # -----------------------------------------------------
        # 1. 保存old policy
        # -----------------------------------------------------

        with torch.no_grad():

            old_dist = self.actor(
                states
            )

            old_probs = (
                old_dist.probs
                .detach()
            )


        old_params = (
            self._flat_params()
            .clone()
        )


        # -----------------------------------------------------
        # 2. Surrogate Objective
        # -----------------------------------------------------

        objective = (
            self._surrogate_objective(
                states,
                actions,
                old_log_probs,
                advantages,
            )
        )


        old_objective = (
            objective
            .detach()
        )


        # -----------------------------------------------------
        # 3. Policy Gradient g
        # -----------------------------------------------------

        policy_gradient = (
            self._flat_grad(
                objective
            )
            .detach()
        )


        # -----------------------------------------------------
        # 4. Natural Gradient Direction
        #
        # solve:
        #
        # Hx = g
        #
        # x ≈ H^-1 g
        # -----------------------------------------------------

        step_direction = (
            self._conjugate_gradient(
                states,
                old_probs,
                policy_gradient,
            )
        )


        # -----------------------------------------------------
        # 5. 根据KL约束计算最大step
        # -----------------------------------------------------

        H_step = (
            self._hessian_vector_product(
                states,
                old_probs,
                step_direction,
            )
        )


        quadratic_term = (
            torch.dot(
                step_direction,
                H_step,
            )
        )


        step_scale = torch.sqrt(
            (
                2.0
                *
                self.max_kl
            )
            /
            (
                quadratic_term
                +
                1e-8
            )
        )


        full_step = (
            step_scale
            *
            step_direction
        )


        # -----------------------------------------------------
        # 6. Backtracking Line Search
        # -----------------------------------------------------

        accepted = False

        final_kl = 0.0

        line_search_step = -1


        for i in range(
            self.backtrack_iters
        ):

            fraction = (
                self.backtrack_coeff
                ** i
            )


            candidate_params = (
                old_params
                +
                fraction
                *
                full_step
            )


            self._set_flat_params(
                candidate_params
            )


            with torch.no_grad():

                new_objective = (
                    self._surrogate_objective(
                        states,
                        actions,
                        old_log_probs,
                        advantages,
                    )
                )


                new_kl = (
                    self._mean_kl(
                        states,
                        old_probs,
                    )
                )


            # 必须同时满足：
            #
            # 1. surrogate提高
            # 2. KL没有超过max_kl

            if (
                new_objective
                >
                old_objective
                and
                new_kl
                <=
                self.max_kl
            ):

                accepted = True

                final_kl = (
                    new_kl.item()
                )

                line_search_step = i

                break


        # 如果所有line search都失败
        # 回到旧policy

        if not accepted:

            self._set_flat_params(
                old_params
            )

            final_kl = 0.0


        return {
            "objective": old_objective.item(),
            "kl": final_kl,
            "line_search_step": line_search_step,
            "accepted": accepted,
        }


    # =========================================================
    # Critic Update
    # =========================================================

    def _update_critic(
        self,
        states,
        returns,
    ):

        critic_loss_value = 0.0


        for _ in range(
            self.critic_updates
        ):

            values = self.critic(
                states
            )


            critic_loss = F.mse_loss(
                values,
                returns,
            )


            self.critic_optimizer.zero_grad()

            critic_loss.backward()

            self.critic_optimizer.step()


            critic_loss_value = (
                critic_loss.item()
            )


        return critic_loss_value


    # =========================================================
    # Full TRPO Update
    # =========================================================

    def update(
        self,
        states,
        actions,
        old_log_probs,
        advantages,
        returns,
    ):

        # Advantage normalization

        advantages = (
            advantages
            -
            advantages.mean()
        ) / (
            advantages.std()
            +
            1e-8
        )


        actor_info = (
            self._update_actor(
                states,
                actions,
                old_log_probs,
                advantages.detach(),
            )
        )


        critic_loss = (
            self._update_critic(
                states,
                returns.detach(),
            )
        )


        actor_info[
            "critic_loss"
        ] = critic_loss


        return actor_info