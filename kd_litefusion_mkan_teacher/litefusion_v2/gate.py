from typing import List

import torch
import torch.nn as nn


def gate_evidence(
    image_feature: torch.Tensor,
    text_feature: torch.Tensor,
    fusion_feature: torch.Tensor,
) -> torch.Tensor:
    if image_feature.shape != text_feature.shape or image_feature.shape != fusion_feature.shape:
        raise ValueError("image_feature, text_feature, and fusion_feature must have identical shapes")
    return torch.cat(
        (image_feature, text_feature, torch.abs(image_feature - text_feature), fusion_feature),
        dim=-1,
    )


class DenseReliabilityGate(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image_feature: torch.Tensor,
        text_feature: torch.Tensor,
        fusion_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(gate_evidence(image_feature, text_feature, fusion_feature))


class GroupedReliabilityGate(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, groups: int, dropout: float):
        super().__init__()
        if feature_dim % groups or hidden % groups:
            raise ValueError("feature_dim and gate_hidden must be divisible by gate_groups")
        self.feature_dim = int(feature_dim)
        self.groups = int(groups)
        self.group_dim = feature_dim // groups
        self.group_hidden = hidden // groups
        self.group_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.group_dim * 4, self.group_hidden),
                    nn.GELU(),
                    nn.LayerNorm(self.group_hidden),
                    nn.Dropout(dropout),
                    nn.Linear(self.group_hidden, self.group_dim),
                    nn.Sigmoid(),
                )
                for _ in range(groups)
            ]
        )

    def forward(
        self,
        image_feature: torch.Tensor,
        text_feature: torch.Tensor,
        fusion_feature: torch.Tensor,
    ) -> torch.Tensor:
        if image_feature.ndim != 2 or image_feature.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected [batch, {self.feature_dim}] image_feature")
        image_groups = image_feature.split(self.group_dim, dim=-1)
        text_groups = text_feature.split(self.group_dim, dim=-1)
        fusion_groups = fusion_feature.split(self.group_dim, dim=-1)
        outputs: List[torch.Tensor] = []
        for index, net in enumerate(self.group_nets):
            image_group = image_groups[index]
            text_group = text_groups[index]
            evidence = torch.cat(
                (image_group, text_group, torch.abs(image_group - text_group), fusion_groups[index]),
                dim=-1,
            )
            outputs.append(net(evidence))
        return torch.cat(outputs, dim=-1)


def build_gate(gate_type: str, feature_dim: int, hidden: int, groups: int, dropout: float) -> nn.Module:
    if gate_type in {"legacy", "bottleneck"}:
        return DenseReliabilityGate(feature_dim=feature_dim, hidden=hidden, dropout=dropout)
    if gate_type == "grouped":
        return GroupedReliabilityGate(
            feature_dim=feature_dim,
            hidden=hidden,
            groups=groups,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported gate_type: {gate_type!r}")
