"""Paper-aligned MKAN-Refine v2 head using real B-spline KAN layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .kan import BSplineKANLinear, kan_regularization
except ImportError:  # direct script/test execution
    from kan import BSplineKANLinear, kan_regularization


@dataclass(frozen=True)
class SplineConfig:
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple[float, float] = (-1.0, 1.0)
    grid_eps: float = 0.02
    normalize_input: bool = True
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0

    def kwargs(self) -> dict:
        return asdict(self)


class KANDualStreamAttention(nn.Module):
    """Symmetric text→vision and vision→text nonlinear refinement."""

    def __init__(self, dim: int = 768, spline: SplineConfig | None = None, share_scorer: bool = True):
        super().__init__()
        spline = spline or SplineConfig()
        kwargs = spline.kwargs()
        self.dim = dim
        self.share_scorer = bool(share_scorer)
        self.text_context = BSplineKANLinear(dim, dim, **kwargs)
        self.vision_context = BSplineKANLinear(dim, dim, **kwargs)
        self.vision_score = BSplineKANLinear(dim, 1, **kwargs)
        self.text_score = self.vision_score if self.share_scorer else BSplineKANLinear(dim, 1, **kwargs)

    def forward(
        self,
        text_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        text_global: torch.Tensor,
        vision_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if text_tokens.ndim != 3 or vision_tokens.ndim != 3:
            raise ValueError("token tensors must have shape [batch,tokens,dim]")
        if text_tokens.shape[0] != vision_tokens.shape[0]:
            raise ValueError("text and vision batch sizes differ")
        if text_tokens.shape[-1] != self.dim or vision_tokens.shape[-1] != self.dim:
            raise ValueError(f"expected token feature dimension {self.dim}")
        text_context = self.text_context(text_global).unsqueeze(1)
        vision_interaction = text_context * vision_tokens.float()
        vision_energy = self.vision_score(vision_interaction).squeeze(-1)
        vision_attention = torch.softmax(vision_energy.float(), dim=1)
        vision_enhanced = torch.sum(vision_attention.unsqueeze(-1) * vision_tokens.float(), dim=1)

        vision_context = self.vision_context(vision_global).unsqueeze(1)
        text_interaction = vision_context * text_tokens.float()
        text_energy = self.text_score(text_interaction).squeeze(-1)
        text_attention = torch.softmax(text_energy.float(), dim=1)
        text_enhanced = torch.sum(text_attention.unsqueeze(-1) * text_tokens.float(), dim=1)
        for name, value in (
            ("vision_enhanced", vision_enhanced),
            ("text_enhanced", text_enhanced),
            ("vision_attention", vision_attention),
            ("text_attention", text_attention),
        ):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite {name}")
        return {
            "vision_enhanced": vision_enhanced,
            "text_enhanced": text_enhanced,
            "vision_attention": vision_attention,
            "text_attention": text_attention,
        }


class MKANPaperHeadV2(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_classes: int = 5,
        classifier_hidden: int = 512,
        dropout: float = 0.3,
        spline: SplineConfig | None = None,
        share_attention_scorer: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.classifier_hidden = classifier_hidden
        self.spline_config = spline or SplineConfig()
        kwargs = self.spline_config.kwargs()
        self.cross_attention = KANDualStreamAttention(
            dim=dim,
            spline=self.spline_config,
            share_scorer=share_attention_scorer,
        )
        self.gate = BSplineKANLinear(dim * 2, dim, **kwargs)
        self.classifier_hidden_layer = BSplineKANLinear(dim, classifier_hidden, **kwargs)
        self.classifier_norm = nn.LayerNorm(classifier_hidden)
        self.classifier_activation = nn.SiLU()
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier_output_layer = BSplineKANLinear(classifier_hidden, num_classes, **kwargs)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        vision_global: torch.Tensor,
        text_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        attention = self.cross_attention(
            text_tokens=text_tokens,
            vision_tokens=vision_tokens,
            text_global=text_global,
            vision_global=vision_global,
        )
        vision_refined = vision_global.float() + attention["vision_enhanced"]
        text_refined = text_global.float() + attention["text_enhanced"]
        gate = torch.sigmoid(self.gate(torch.cat([vision_refined, text_refined], dim=-1)))
        fused = (1.0 - gate) * vision_refined + gate * text_refined
        hidden = self.classifier_hidden_layer(fused)
        hidden = self.classifier_dropout(self.classifier_activation(self.classifier_norm(hidden)))
        logits = self.classifier_output_layer(hidden)
        for name, value in (("gate", gate), ("fused", fused), ("logits", logits)):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite {name}")
        return {
            "logits": logits,
            "feature": fused,
            "gate": gate,
            "vision_refined": vision_refined,
            "text_refined": text_refined,
            "vision_attention": attention["vision_attention"],
            "text_attention": attention["text_attention"],
        }

    def regularization_loss(self, activation: float = 1.0, entropy: float = 1.0):
        return kan_regularization(self.modules(), activation, entropy).to(
            next(self.parameters()).device
        )

    @torch.no_grad()
    def update_grids(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        vision_global: torch.Tensor,
        text_global: torch.Tensor,
        margin: float = 0.01,
    ) -> list[dict]:
        captured: dict[int, list[torch.Tensor]] = {}
        handles = []
        unique_modules = []
        seen = set()
        for module in self.modules():
            if isinstance(module, BSplineKANLinear) and id(module) not in seen:
                seen.add(id(module))
                unique_modules.append(module)
                captured[id(module)] = []
                handles.append(
                    module.register_forward_pre_hook(
                        lambda current, inputs, module_id=id(module): captured[module_id].append(
                            inputs[0].detach().reshape(-1, current.in_features).float()
                        )
                    )
                )
        self.forward(vision_tokens, text_tokens, vision_global, text_global)
        for handle in handles:
            handle.remove()
        reports = []
        for index, module in enumerate(unique_modules):
            samples = torch.cat(captured[id(module)], dim=0)
            report = module.update_grid(samples, margin=margin)
            reports.append({"module_index": index, "module": repr(module), **report})
        return reports


class CachedFeatureMKAN(nn.Module):
    """Training wrapper for frozen, precomputed CLIP token/global features."""

    def __init__(self, head: MKANPaperHeadV2):
        super().__init__()
        self.head = head

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.head(
            vision_tokens=batch["vision_tokens"],
            text_tokens=batch["text_tokens"],
            vision_global=batch["vision_global"],
            text_global=batch["text_global"],
        )
