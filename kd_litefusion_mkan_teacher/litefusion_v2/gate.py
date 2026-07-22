import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # A grouped 1x1 convolution is exactly a bank of independent linear
        # layers. It preserves group-specific dynamic reliability while
        # replacing 32 Python-loop iterations and dozens of tiny CUDA kernels
        # with two vectorized grouped kernels.
        self.input_projection = nn.Conv1d(
            in_channels=feature_dim * 4,
            out_channels=hidden,
            kernel_size=1,
            groups=groups,
        )
        self.norm_weight = nn.Parameter(torch.ones(groups, self.group_hidden))
        self.norm_bias = nn.Parameter(torch.zeros(groups, self.group_hidden))
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Conv1d(
            in_channels=hidden,
            out_channels=feature_dim,
            kernel_size=1,
            groups=groups,
        )

    def forward(
        self,
        image_feature: torch.Tensor,
        text_feature: torch.Tensor,
        fusion_feature: torch.Tensor,
    ) -> torch.Tensor:
        if image_feature.ndim != 2 or image_feature.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected [batch, {self.feature_dim}] image_feature")
        if image_feature.shape != text_feature.shape or image_feature.shape != fusion_feature.shape:
            raise ValueError("image_feature, text_feature, and fusion_feature must have identical shapes")
        batch_size = image_feature.shape[0]
        image_groups = image_feature.reshape(batch_size, self.groups, self.group_dim)
        text_groups = text_feature.reshape(batch_size, self.groups, self.group_dim)
        fusion_groups = fusion_feature.reshape(batch_size, self.groups, self.group_dim)
        evidence = torch.stack(
            (image_groups, text_groups, torch.abs(image_groups - text_groups), fusion_groups),
            dim=2,
        ).reshape(batch_size, self.groups, self.group_dim * 4)
        evidence = evidence.reshape(batch_size, self.feature_dim * 4, 1)
        hidden = self.input_projection(evidence).reshape(batch_size, self.groups, self.group_hidden)
        hidden = F.gelu(hidden)
        hidden = F.layer_norm(hidden, (self.group_hidden,))
        hidden = hidden * self.norm_weight.unsqueeze(0) + self.norm_bias.unsqueeze(0)
        hidden = self.dropout(hidden).reshape(batch_size, -1, 1)
        return torch.sigmoid(self.output_projection(hidden).reshape(batch_size, self.feature_dim))


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
