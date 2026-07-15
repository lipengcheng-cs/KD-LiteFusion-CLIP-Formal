import os
from typing import Dict, Union

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankCrossModalFusion(nn.Module):
    def __init__(self, dim: int, rank: int = 32, dropout: float = 0.1):
        super().__init__()
        self.vision_down = nn.Linear(dim, rank, bias=False)
        self.text_down = nn.Linear(dim, rank, bias=False)
        self.interaction_up = nn.Linear(rank, dim)
        self.residual = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vision_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        low_rank = self.vision_down(vision_feat) * self.text_down(text_feat)
        interaction = self.interaction_up(low_rank)
        residual = self.residual(torch.cat([vision_feat, text_feat], dim=-1))
        return self.norm(self.dropout(interaction) + residual)


class ReliabilityAwareGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        vision_feat: torch.Tensor,
        text_feat: torch.Tensor,
        fusion_feat: torch.Tensor,
    ) -> torch.Tensor:
        evidence = torch.cat(
            [vision_feat, text_feat, torch.abs(vision_feat - text_feat), fusion_feat],
            dim=-1,
        )
        return self.net(evidence)


class KDLiteFusionCLIP(nn.Module):
    def __init__(
        self,
        clip_model_path: str,
        num_classes: int,
        rank: int = 32,
        dropout: float = 0.2,
        freeze_clip: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        clip_model_path = os.path.abspath(os.path.expanduser(clip_model_path))
        if not os.path.isfile(clip_model_path):
            raise FileNotFoundError(
                f"OpenAI CLIP checkpoint not found: {clip_model_path}. "
                "Expected /home/lpc/.cache/clip/ViT-L-14-336px.pt; network downloads are disabled."
            )
        try:
            self.clip, self.preprocess = clip.load(clip_model_path, device=device, jit=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local OpenAI CLIP checkpoint {clip_model_path!r}: {exc}"
            ) from exc
        dim = int(self.clip.text_projection.shape[1])
        if dim != 768:
            raise ValueError(f"Expected OpenAI CLIP ViT-L/14@336px feature dimension 768, got {dim}")
        self.dim = dim
        self.freeze_clip = freeze_clip
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
            self.clip.eval()

        self.fusion = LowRankCrossModalFusion(dim=dim, rank=rank, dropout=dropout)
        self.gate = ReliabilityAwareGate(dim=dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def encode_clip(
        self,
        text_tokens: torch.Tensor,
        images: torch.Tensor,
    ):
        context = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with context:
            vision_feat = self.clip.encode_image(images)
            text_feat = self.clip.encode_text(text_tokens)
            vision_feat = F.normalize(vision_feat.float(), dim=-1)
            text_feat = F.normalize(text_feat.float(), dim=-1)
        return vision_feat, text_feat

    def forward(
        self,
        text_tokens: torch.Tensor,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        vision_feat, text_feat = self.encode_clip(text_tokens, images)
        fusion_feat = self.fusion(vision_feat, text_feat)
        gate = self.gate(vision_feat, text_feat, fusion_feat)
        fused = fusion_feat + gate * text_feat + (1.0 - gate) * vision_feat
        fused = F.normalize(fused, dim=-1)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "feature": fused,
            "gate": gate,
            "vision_feature": vision_feat,
            "text_feature": text_feat,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    def student_state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value for key, value in self.state_dict().items() if not key.startswith("clip.")}

    def load_student_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_student = [key for key in incompatible.missing_keys if not key.startswith("clip.")]
        if unexpected or missing_student:
            raise RuntimeError(
                f"Invalid student checkpoint; missing={missing_student}, unexpected={unexpected}"
            )
