"""Numerically guarded B-spline Kolmogorov–Arnold linear layer.

The edge function follows the paper form

    phi(x) = w_b * SiLU(x) + w_s * sum_k c_k B_k(x)

instead of approximating the spline branch with another ordinary Linear layer.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineKANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        grid_eps: float = 0.02,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        normalize_input: bool = True,
        check_finite: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if grid_size < 2:
            raise ValueError("grid_size must be >=2")
        if spline_order < 1:
            raise ValueError("spline_order must be >=1")
        if not grid_range[0] < grid_range[1]:
            raise ValueError("grid_range must be strictly increasing")
        if not 0.0 <= grid_eps <= 1.0:
            raise ValueError("grid_eps must be in [0,1]")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.scale_noise = float(scale_noise)
        self.scale_base = float(scale_base)
        self.scale_spline = float(scale_spline)
        self.enable_standalone_scale_spline = bool(enable_standalone_scale_spline)
        self.grid_eps = float(grid_eps)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.normalize_input = bool(normalize_input)
        self.check_finite = bool(check_finite)

        step = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        knots = (
            torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=torch.float32)
            * step
            + self.grid_range[0]
        )
        self.register_buffer("grid", knots.expand(self.in_features, -1).contiguous())
        self.input_norm = (
            nn.LayerNorm(self.in_features, elementwise_affine=True)
            if self.normalize_input
            else nn.Identity()
        )
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.base_bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, self.grid_size + self.spline_order)
        )
        if self.enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(self.out_features, self.in_features))
        else:
            self.register_parameter("spline_scaler", None)
        self.reset_parameters()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.spline_scaler is None:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(x.float())
        # LayerNorm alone is unbounded and can move activations outside the
        # finite knot support, silently turning the spline branch into zero.
        # Map normalized activations into the configured interior knot range.
        if self.normalize_input:
            low, high = self.grid_range
            center = 0.5 * (low + high)
            radius = 0.5 * (high - low)
            normalized = center + radius * torch.tanh(normalized)
        return normalized

    def b_splines(self, x: torch.Tensor, *, already_normalized: bool = False) -> torch.Tensor:
        """Return B-spline bases with shape ``[..., in_features, n_basis]``."""
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}, got {tuple(x.shape)}")
        original_shape = x.shape
        flat = x.reshape(-1, self.in_features).float()
        if not already_normalized:
            flat = self._normalize(flat)
        grid = self.grid.float()
        # Keep the rightmost boundary inside the final half-open interval.
        upper = grid[:, -1] - torch.finfo(flat.dtype).eps
        flat = torch.minimum(flat, upper.unsqueeze(0))
        x_expanded = flat.unsqueeze(-1)
        bases = ((x_expanded >= grid[:, :-1]) & (x_expanded < grid[:, 1:])).to(flat.dtype)
        eps = torch.finfo(flat.dtype).eps
        for degree in range(1, self.spline_order + 1):
            left_denominator = grid[:, degree:-1] - grid[:, : -(degree + 1)]
            right_denominator = grid[:, degree + 1 :] - grid[:, 1:-degree]
            left = (x_expanded - grid[:, : -(degree + 1)]) / left_denominator.clamp_min(eps)
            right = (grid[:, degree + 1 :] - x_expanded) / right_denominator.clamp_min(eps)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        expected = self.grid_size + self.spline_order
        if bases.shape[-1] != expected:
            raise RuntimeError(f"basis construction produced {bases.shape[-1]} functions, expected {expected}")
        if self.check_finite and not torch.isfinite(bases).all():
            raise FloatingPointError("non-finite B-spline basis")
        return bases.reshape(*original_shape[:-1], self.in_features, expected).contiguous()

    def curve2coeff(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        """Fit per-edge spline coefficients to samples.

        ``x`` is ``[samples,in_features]`` and ``y`` is
        ``[samples,in_features,out_features]``.
        """
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("x must have shape [samples,in_features]")
        if y.shape != (x.shape[0], self.in_features, self.out_features):
            raise ValueError(
                f"y must have shape {(x.shape[0], self.in_features, self.out_features)}, got {tuple(y.shape)}"
            )
        basis = self.b_splines(x, already_normalized=already_normalized).permute(1, 0, 2)
        targets = y.float().permute(1, 0, 2)
        solution = torch.linalg.lstsq(basis, targets, driver="gels").solution
        coefficients = solution.permute(2, 0, 1).contiguous()
        if self.check_finite and not torch.isfinite(coefficients).all():
            raise FloatingPointError("non-finite spline coefficients")
        return coefficients

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.base_weight.mul_(self.scale_base)
        if self.base_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.base_bias, -bound, bound)
        # Small direct coefficient noise is a stable curve initialization and avoids
        # a very large batched least-squares solve for layers such as 1536→768.
        coefficient_std = self.scale_noise / math.sqrt(
            self.in_features * (self.grid_size + self.spline_order)
        )
        nn.init.normal_(self.spline_weight, mean=0.0, std=coefficient_std)
        if self.spline_scaler is None:
            with torch.no_grad():
                self.spline_weight.mul_(self.scale_spline)
        else:
            nn.init.constant_(self.spline_scaler, self.scale_spline)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}, got {tuple(x.shape)}")
        leading_shape = x.shape[:-1]
        flat = self._normalize(x.reshape(-1, self.in_features))
        base_output = F.linear(F.silu(flat), self.base_weight.float(), None if self.base_bias is None else self.base_bias.float())
        basis = self.b_splines(flat, already_normalized=True).reshape(flat.shape[0], -1)
        spline_output = F.linear(basis, self.scaled_spline_weight.float().reshape(self.out_features, -1))
        output = (base_output + spline_output).reshape(*leading_shape, self.out_features)
        if self.check_finite and not torch.isfinite(output).all():
            raise FloatingPointError("non-finite BSplineKANLinear output")
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> dict:
        """Adapt knots to the activation distribution while preserving the spline curve."""
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}, got {tuple(x.shape)}")
        flat = self._normalize(x.reshape(-1, self.in_features))
        if flat.shape[0] < self.grid_size + 1:
            raise ValueError(f"grid update needs at least {self.grid_size + 1} samples")
        if not torch.isfinite(flat).all():
            raise FloatingPointError("non-finite grid-update input")

        old_basis = self.b_splines(flat, already_normalized=True)
        unreduced = torch.einsum("bic,oic->bio", old_basis, self.scaled_spline_weight.float())
        sorted_x = torch.sort(flat, dim=0).values
        indices = torch.linspace(0, flat.shape[0] - 1, self.grid_size + 1, device=flat.device).round().long()
        adaptive = sorted_x[indices]
        data_min, data_max = sorted_x[0], sorted_x[-1]
        step = (data_max - data_min + 2 * margin) / self.grid_size
        step = step.clamp_min(torch.finfo(flat.dtype).eps * 32)
        uniform = (
            torch.arange(self.grid_size + 1, device=flat.device, dtype=flat.dtype).unsqueeze(1) * step
            + data_min
            - margin
        )
        interior = self.grid_eps * uniform + (1.0 - self.grid_eps) * adaptive
        left = interior[:1] - step * torch.arange(
            self.spline_order, 0, -1, device=flat.device, dtype=flat.dtype
        ).unsqueeze(1)
        right = interior[-1:] + step * torch.arange(
            1, self.spline_order + 1, device=flat.device, dtype=flat.dtype
        ).unsqueeze(1)
        new_grid = torch.cat([left, interior, right], dim=0).T.contiguous()
        if not torch.all(new_grid[:, 1:] > new_grid[:, :-1]):
            raise FloatingPointError("adaptive grid is not strictly increasing")
        old_grid = self.grid.detach().clone()
        self.grid.copy_(new_grid)
        scaled_coefficients = self.curve2coeff(flat, unreduced, already_normalized=True)
        if self.spline_scaler is None:
            self.spline_weight.copy_(scaled_coefficients)
        else:
            scaler = self.spline_scaler.detach()
            safe = torch.where(
                scaler.abs() < 1e-8,
                torch.where(scaler >= 0, torch.full_like(scaler, 1e-8), torch.full_like(scaler, -1e-8)),
                scaler,
            )
            self.spline_weight.copy_(scaled_coefficients / safe.unsqueeze(-1))
        return {
            "old_min": float(old_grid.min().item()),
            "old_max": float(old_grid.max().item()),
            "new_min": float(self.grid.min().item()),
            "new_max": float(self.grid.max().item()),
            "samples": int(flat.shape[0]),
        }

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        edge_strength = self.scaled_spline_weight.abs().mean(dim=-1)
        activation = edge_strength.sum()
        probabilities = edge_strength / activation.clamp_min(torch.finfo(edge_strength.dtype).eps)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        return regularize_activation * activation + regularize_entropy * entropy

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}, "
            f"normalize_input={self.normalize_input}"
        )


def kan_regularization(modules: Iterable[nn.Module], activation: float = 1.0, entropy: float = 1.0):
    losses = [
        module.regularization_loss(activation, entropy)
        for module in modules
        if isinstance(module, BSplineKANLinear)
    ]
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).sum()
