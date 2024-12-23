from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer


def adamw_step(
    p: nn.Parameter,
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    step_factor: torch.Tensor,
):
    if p.grad is None:
        return

    beta1, beta2 = betas

    # Perform step weight decay.
    p.mul_(1 - step_factor * (lr * weight_decay))

    # Decay the first and second moment running average coefficient.
    exp_avg.lerp_(p.grad, step_factor * (1 - beta1))
    exp_avg_sq.mul_(1 - step_factor * (1 - beta2))
    exp_avg_sq.add_(step_factor * p.grad * p.grad, alpha=1 - beta2)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    step_size = lr / bias_correction1

    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

    update = -step_size * torch.div(exp_avg, denom)
    update.mul_(step_factor)
    p.add_(update)


class AdamW(Optimizer):
    """
    An implementation of the AdamW optimizer.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        foreach: Optional[bool] = None,
        fused: Optional[bool] = None,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=foreach, fused=fused
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                adamw_step(
                    p,
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    exp_avg=state["exp_avg"],
                    exp_avg_sq=state["exp_avg_sq"],
                    step=state["step"],
                    step_factor=torch.tensor(1.0, device=p.device),
                )


class SkipStepAdamW(SkipStepOptimizer):
    """
    A "skip step" version of :class:`AdamW`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        foreach: Optional[bool] = None,
        fused: Optional[bool] = None,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
    ) -> None:
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=foreach, fused=fused
        )
        super().__init__(
            params,
            defaults,
            rolling_interval_length=rolling_interval_length,
            sigma_factor=sigma_factor,
        )
        self._step_skipped: Optional[torch.Tensor] = None

    @property
    def step_skipped(self) -> torch.Tensor:
        if self._step_skipped is not None:
            return self._step_skipped
        else:
            return torch.tensor(0.0)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                adamw_step(
                    p,
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    exp_avg=state["exp_avg"],
                    exp_avg_sq=state["exp_avg_sq"],
                    step=state["step"],
                    step_factor=step_factor,
                )


@dataclass
class AdamWConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
    """
    Configuration class for building an :class:`torch.optim.AdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    foreach: Optional[bool] = None
    fused: Optional[bool] = None

    @classmethod
    def optimizer(cls) -> Type[AdamW]:
        return AdamW


@dataclass
class SkipStepAdamWConfig(OptimConfig):
    """
    Configuration class for building a :class:`SkipStepAdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    foreach: Optional[bool] = None
    fused: Optional[bool] = None
    rolling_interval_length: int = 128
    sigma_factor: int = 6

    @classmethod
    def optimizer(cls) -> Type[SkipStepAdamW]:
        return SkipStepAdamW
