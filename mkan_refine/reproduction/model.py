"""MKAN-Refine source-implementation reproduction on top of OpenAI CLIP.

This module intentionally mirrors the model topology in the supplied
``inference.py``.  Its ``KANLinear`` is therefore the two-branch nonlinear
layer from that source file, not a claim of a faithful B-spline KAN.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NATIVE_LABEL_TO_ID = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "not_humanitarian": 2,
    "other_relevant_information": 3,
    "rescue_volunteering_or_donation_effort": 4,
}

STUDENT_LABEL_TO_ID = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "rescue_volunteering_or_donation_effort": 2,
    "other_relevant_information": 3,
    "not_humanitarian": 4,
}


class KANLinear(nn.Module):
    """The nonlinear layer exactly as supplied in the available source."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.spline = nn.Linear(in_features, out_features)
        self.scale_base = nn.Parameter(torch.ones(out_features) * 0.1)
        self.scale_spline = nn.Parameter(torch.ones(out_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(F.silu(x))
        spline_out = self.spline(x * torch.sigmoid(x))
        return base_out * self.scale_base + spline_out * self.scale_spline


class KANDualAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.text_pool = nn.Linear(dim, 1)
        self.score = KANLinear(dim, 1)

    def forward(
        self, text_tokens: torch.Tensor, vision_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_weights = torch.softmax(self.text_pool(text_tokens), dim=1)
        text_context = (text_weights * text_tokens).sum(dim=1, keepdim=True)

        inter_vision = text_context * vision_tokens
        vision_weights = torch.softmax(self.score(inter_vision), dim=1)
        vision_feat = (vision_weights * vision_tokens).sum(dim=1)

        vision_context = vision_tokens.mean(dim=1, keepdim=True)
        inter_text = vision_context * text_tokens
        text_weights_refined = torch.softmax(self.score(inter_text), dim=1)
        text_feat = (text_weights_refined * text_tokens).sum(dim=1)
        return vision_feat, text_feat


class MKANHead(nn.Module):
    """Trainable MKAN fusion/gating/classification head for cached CLIP features."""

    def __init__(self, dim: int = 768, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.cross_attn = KANDualAttention(dim)
        self.gate = KANLinear(dim * 2, dim)
        self.classifier = nn.Sequential(
            KANLinear(dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            KANLinear(512, num_classes),
        )

    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        vision_global: torch.Tensor,
        text_global: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Cached features are stored in float16 to save space; the head trains in fp32.
        vision_tokens = vision_tokens.float()
        text_tokens = text_tokens.float()
        vision_global = vision_global.float()
        text_global = text_global.float()

        vision_enhanced, text_enhanced = self.cross_attn(text_tokens, vision_tokens)
        vision_final = vision_global + vision_enhanced
        text_final = text_global + text_enhanced
        concat_feat = torch.cat([vision_final, text_final], dim=-1)
        gate = torch.sigmoid(self.gate(concat_feat))
        fused = vision_final + gate * (text_final - vision_final)
        logits = self.classifier(fused)
        return {"logits": logits, "feature": fused, "gate": gate}


class OpenAIClipTokenEncoder(nn.Module):
    """Expose token and global projected features from an OpenAI CLIP model."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model
        self.clip.requires_grad_(False)
        self.clip.eval()

    @torch.no_grad()
    def encode(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.clip
        visual = model.visual
        dtype = model.dtype

        image = pixel_values.to(dtype=dtype)
        x = visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype)
        class_token = class_token + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        vision_tokens = x @ visual.proj if visual.proj is not None else x
        vision_global = vision_tokens[:, 0]

        text = model.token_embedding(input_ids).to(dtype)
        text = text + model.positional_embedding.to(dtype)
        text = text.permute(1, 0, 2)
        text = model.transformer(text)
        text = text.permute(1, 0, 2)
        text = model.ln_final(text).to(dtype)
        text_tokens = text @ model.text_projection
        eot_positions = input_ids.argmax(dim=-1)
        text_global = text_tokens[torch.arange(text_tokens.shape[0], device=text_tokens.device), eot_positions]

        return vision_tokens, text_tokens, vision_global, text_global
