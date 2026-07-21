import torch
import torch.nn as nn


class LiteFusionV2Classifier(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, final_feature: torch.Tensor) -> torch.Tensor:
        if final_feature.ndim != 2 or final_feature.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected [batch, {self.feature_dim}] final_feature")
        return self.net(final_feature)
