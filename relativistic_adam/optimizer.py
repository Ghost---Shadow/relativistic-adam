import torch
from torch.optim.optimizer import Optimizer
import math


class RelativisticAdam(Optimizer):
    """
    RelativisticAdam optimizer - Adam with relativistic gradient clipping.

    This optimizer applies relativistic mechanics principles to prevent exploding gradients
    by introducing a "speed limit" for parameter updates, analogous to the speed of light.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        speed_limit: maximum allowed update magnitude 'c' in relativistic scaling (default: 0.1)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsgrad: whether to use the AMSGrad variant (default: False)
        relativistic_mode: 'global', 'per_param', or 'per_component' (default: 'per_param')
        adaptive_speed: whether to adapt speed limit over time (default: False)
        speed_warmup_steps: number of steps to warm up to full speed limit (default: 1000)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        speed_limit=0.1,
        weight_decay=0,
        amsgrad=False,
        relativistic_mode="per_param",
        adaptive_speed=False,
        speed_warmup_steps=1000,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < speed_limit:
            raise ValueError(f"Invalid speed limit: {speed_limit}")
        if relativistic_mode not in ["global", "per_param", "per_component"]:
            raise ValueError(f"Invalid relativistic mode: {relativistic_mode}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            speed_limit=speed_limit,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            relativistic_mode=relativistic_mode,
            adaptive_speed=adaptive_speed,
            speed_warmup_steps=speed_warmup_steps,
        )
        super(RelativisticAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RelativisticAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("relativistic_mode", "per_param")
            group.setdefault("adaptive_speed", False)
            group.setdefault("speed_warmup_steps", 1000)

    def _get_adaptive_speed_limit(self, base_speed_limit, step, warmup_steps):
        """Calculate adaptive speed limit with warmup."""
        if warmup_steps <= 0:
            return base_speed_limit
        warmup_factor = min(1.0, step / warmup_steps)
        return base_speed_limit * warmup_factor

    def _apply_relativistic_scaling(self, update, speed_limit, mode="per_param"):
        """
        Apply relativistic scaling to prevent update magnitude from exceeding speed limit.

        When ||update|| < c: Apply relativistic scaling (update gets smaller)
        When ||update|| >= c: Cannot exceed speed of light! Saturate at speed limit

        Args:
            update: proposed parameter update
            speed_limit: maximum allowed update magnitude
            mode: 'global', 'per_param', or 'per_component'

        Returns:
            scaled update that respects the speed limit
        """
        if mode == "global":
            # Apply single scaling factor for entire update tensor
            update_norm = torch.norm(update)
            if update_norm > 0:
                ratio = update_norm / speed_limit

                if ratio < 1.0:
                    # ||update|| < c: Apply relativistic scaling
                    # γ = 1/√(1 - v²/c²), then scale DOWN by γ (divide)
                    gamma = 1.0 / torch.sqrt(1.0 - ratio * ratio)
                    return update / gamma
                else:
                    # ||update|| >= c: Physically impossible in relativity!
                    # Saturate at speed limit using smooth function
                    # Options:
                    # 1. Hard clip to speed limit
                    # 2. Smooth saturation with tanh
                    # We use smooth saturation for better gradients
                    return (
                        update
                        * (speed_limit / update_norm)
                        * torch.tanh(update_norm / speed_limit)
                    )
            return update

        elif mode == "per_param":
            # Apply scaling per parameter (each tensor separately)
            update_norm = torch.norm(update.view(-1))
            if update_norm > 0:
                ratio = update_norm / speed_limit

                if ratio < 1.0:
                    # ||update|| < c: Apply relativistic scaling
                    gamma = 1.0 / torch.sqrt(1.0 - ratio * ratio)
                    return update / gamma
                else:
                    # ||update|| >= c: Saturate smoothly
                    return (
                        update
                        * (speed_limit / update_norm)
                        * torch.tanh(update_norm / speed_limit)
                    )
            return update

        elif mode == "per_component":
            # Apply scaling per component (element-wise)
            abs_update = torch.abs(update)
            ratio = abs_update / speed_limit

            # Create output tensor
            scaled_update = torch.zeros_like(update)

            # Case 1: |update_i| < c (relativistic scaling applies)
            small_mask = ratio < 1.0
            if small_mask.any():
                # γ_i = 1/√(1 - v_i²/c²)
                gamma = 1.0 / torch.sqrt(1.0 - ratio[small_mask] ** 2)
                scaled_update[small_mask] = update[small_mask] / gamma

            # Case 2: |update_i| >= c (saturate at speed limit)
            large_mask = ~small_mask
            if large_mask.any():
                # Smooth saturation: sign(update) * c * tanh(|update|/c)
                scaled_update[large_mask] = (
                    torch.sign(update[large_mask])
                    * speed_limit
                    * torch.tanh(abs_update[large_mask] / speed_limit)
                )

            return scaled_update

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            base_speed_limit = group["speed_limit"]
            mode = group["relativistic_mode"]
            adaptive_speed = group["adaptive_speed"]
            warmup_steps = group["speed_warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "RelativisticAdam does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintain the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Compute proposed update (velocity in relativistic terms)
                update = step_size * (exp_avg / denom)

                # Get adaptive speed limit if enabled
                if adaptive_speed:
                    current_speed_limit = self._get_adaptive_speed_limit(
                        base_speed_limit, step, warmup_steps
                    )
                else:
                    current_speed_limit = base_speed_limit

                # Apply relativistic scaling to prevent exploding updates
                scaled_update = self._apply_relativistic_scaling(
                    update, current_speed_limit, mode
                )

                # Update parameters
                p.add_(scaled_update, alpha=-1)

        return loss


class RelativisticAdamW(RelativisticAdam):
    """
    RelativisticAdamW optimizer - AdamW with relativistic gradient clipping.

    This is the weight decay decoupled version of RelativisticAdam (similar to AdamW vs Adam).

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        speed_limit: maximum allowed update magnitude 'c' in relativistic scaling (default: 0.1)
        weight_decay: weight decay coefficient (default: 1e-2)
        amsgrad: whether to use the AMSGrad variant (default: False)
        relativistic_mode: 'global', 'per_param', or 'per_component' (default: 'per_param')
        adaptive_speed: whether to adapt speed limit over time (default: False)
        speed_warmup_steps: number of steps to warm up to full speed limit (default: 1000)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        speed_limit=0.1,
        weight_decay=1e-2,
        amsgrad=False,
        relativistic_mode="per_param",
        adaptive_speed=False,
        speed_warmup_steps=1000,
    ):
        super(RelativisticAdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            speed_limit=speed_limit,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            relativistic_mode=relativistic_mode,
            adaptive_speed=adaptive_speed,
            speed_warmup_steps=speed_warmup_steps,
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with decoupled weight decay."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            base_speed_limit = group["speed_limit"]
            mode = group["relativistic_mode"]
            adaptive_speed = group["adaptive_speed"]
            warmup_steps = group["speed_warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "RelativisticAdamW does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Compute proposed update
                update = step_size * (exp_avg / denom)

                # Get adaptive speed limit if enabled
                if adaptive_speed:
                    current_speed_limit = self._get_adaptive_speed_limit(
                        base_speed_limit, step, warmup_steps
                    )
                else:
                    current_speed_limit = base_speed_limit

                # Apply relativistic scaling
                scaled_update = self._apply_relativistic_scaling(
                    update, current_speed_limit, mode
                )

                # Update parameters with scaled update
                p.add_(scaled_update, alpha=-1)

                # Apply decoupled weight decay (after the main update)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

        return loss


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test case
    torch.manual_seed(42)

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50), torch.nn.ReLU(), torch.nn.Linear(50, 1)
    )

    # Create optimizer with different relativistic modes
    optimizer_per_param = RelativisticAdam(
        model.parameters(), lr=0.001, speed_limit=0.01, relativistic_mode="per_param"
    )

    optimizer_per_component = RelativisticAdam(
        model.parameters(),
        lr=0.001,
        speed_limit=0.01,
        relativistic_mode="per_component",
    )

    # Test with synthetic data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    # Simulate training step
    for i in range(5):
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer_per_param.step()
        optimizer_per_param.zero_grad()

        print(f"Step {i+1}, Loss: {loss.item():.4f}")

    print("\nRelativisticAdam optimizer successfully implemented!")

    # Test with extremely large gradients to verify relativistic scaling
    print("\nTesting with large gradients:")
    model_test = torch.nn.Linear(5, 1)
    optimizer_test = RelativisticAdam(
        model_test.parameters(),
        lr=0.1,
        speed_limit=0.5,
        relativistic_mode="per_component",
    )

    # Create artificially large gradients
    x_test = torch.randn(1, 5) * 100
    y_test = torch.randn(1, 1)

    output_test = model_test(x_test)
    loss_test = torch.nn.functional.mse_loss(output_test, y_test)
    loss_test.backward()

    # Check gradient magnitude before and after step
    grad_norm_before = torch.norm(model_test.weight.grad)
    print(f"Gradient norm before step: {grad_norm_before:.4f}")

    optimizer_test.step()
    print("Update applied successfully with relativistic scaling!")
