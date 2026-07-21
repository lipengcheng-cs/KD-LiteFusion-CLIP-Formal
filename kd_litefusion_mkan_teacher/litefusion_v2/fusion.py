import torch
import torch.nn as nn


class LiteFusionV2Fusion(nn.Module):
    """Low-rank interaction plus a configurable low-rank residual projection."""

    def __init__(self, feature_dim: int, interaction_rank: int, residual_rank: int, dropout: float):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.interaction_rank = int(interaction_rank)
        self.residual_rank = int(residual_rank)
        self.image_interaction_down = nn.Linear(feature_dim, interaction_rank, bias=False)
        self.text_interaction_down = nn.Linear(feature_dim, interaction_rank, bias=False)
        self.interaction_up = nn.Linear(interaction_rank, feature_dim)
        self.residual_down = nn.Linear(feature_dim * 2, residual_rank, bias=False)
        self.residual_up = nn.Linear(residual_rank, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, image_feature: torch.Tensor, text_feature: torch.Tensor) -> torch.Tensor:
        self._validate(image_feature, text_feature)
        interaction = self.image_interaction_down(image_feature) * self.text_interaction_down(text_feature)
        interaction = self.interaction_up(interaction)
        residual = self.residual_up(self.residual_down(torch.cat((image_feature, text_feature), dim=-1)))
        return self.norm(self.dropout(interaction) + residual)

    def _validate(self, image_feature: torch.Tensor, text_feature: torch.Tensor) -> None:
        if image_feature.shape != text_feature.shape:
            raise ValueError(
                f"image/text feature shape mismatch: {tuple(image_feature.shape)} vs {tuple(text_feature.shape)}"
            )
        if image_feature.ndim != 2 or image_feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected [batch, {self.feature_dim}] features, got {tuple(image_feature.shape)}"
            )
