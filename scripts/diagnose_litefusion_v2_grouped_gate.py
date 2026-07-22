#!/usr/bin/env python3
"""Verify that the vectorized grouped gate is equivalent to the legacy loop."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kd_litefusion_mkan_teacher.litefusion_v2.gate import GroupedReliabilityGate


class LegacyGroupedReliabilityGate(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, groups: int, dropout: float):
        super().__init__()
        if feature_dim % groups or hidden % groups:
            raise ValueError("feature_dim and hidden must be divisible by groups")
        self.feature_dim = feature_dim
        self.groups = groups
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

    def forward(self, image_feature, text_feature, fusion_feature):
        image_groups = image_feature.split(self.group_dim, dim=-1)
        text_groups = text_feature.split(self.group_dim, dim=-1)
        fusion_groups = fusion_feature.split(self.group_dim, dim=-1)
        gates = []
        for image_group, text_group, fusion_group, net in zip(
            image_groups, text_groups, fusion_groups, self.group_nets
        ):
            evidence = torch.cat(
                (
                    image_group,
                    text_group,
                    torch.abs(image_group - text_group),
                    fusion_group,
                ),
                dim=-1,
            )
            gates.append(net(evidence))
        return torch.cat(gates, dim=-1)


def copy_legacy_weights(legacy, vectorized):
    with torch.no_grad():
        for group_index, net in enumerate(legacy.group_nets):
            first = net[0]
            norm = net[2]
            second = net[4]
            hidden_start = group_index * vectorized.group_hidden
            hidden_end = hidden_start + vectorized.group_hidden
            feature_start = group_index * vectorized.group_dim
            feature_end = feature_start + vectorized.group_dim
            vectorized.input_projection.weight[hidden_start:hidden_end, :, 0].copy_(first.weight)
            vectorized.input_projection.bias[hidden_start:hidden_end].copy_(first.bias)
            vectorized.norm_weight[group_index].copy_(norm.weight)
            vectorized.norm_bias[group_index].copy_(norm.bias)
            vectorized.output_projection.weight[feature_start:feature_end, :, 0].copy_(second.weight)
            vectorized.output_projection.bias[feature_start:feature_end].copy_(second.bias)


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "grouped_gate_equivalence.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {result_path}")

    torch.manual_seed(args.seed)
    settings = {"feature_dim": 768, "hidden": 64, "groups": 32, "dropout": 0.0}
    legacy = LegacyGroupedReliabilityGate(**settings).eval()
    vectorized = GroupedReliabilityGate(**settings).eval()
    copy_legacy_weights(legacy, vectorized)

    traces = []
    maximum_error = 0.0
    for batch_size in (1, 8):
        image = torch.randn(batch_size, settings["feature_dim"])
        text = torch.randn_like(image)
        fusion = torch.randn_like(image)
        with torch.inference_mode():
            expected = legacy(image, text, fusion)
            actual = vectorized(image, text, fusion)
        error = float((expected - actual).abs().max().item())
        maximum_error = max(maximum_error, error)
        traces.append(
            {
                "batch_size": batch_size,
                "input_shape": list(image.shape),
                "output_shape": list(actual.shape),
                "finite": bool(torch.isfinite(actual).all()),
                "max_abs_error": error,
            }
        )

    legacy_parameters = parameter_count(legacy)
    vectorized_parameters = parameter_count(vectorized)
    # Conv1d and Linear may accumulate FP32 products in a different order.
    # A 1e-5 absolute tolerance is tight relative to sigmoid outputs while
    # allowing the expected last-bit kernel difference.
    tolerance = 1e-5
    payload = {
        "seed": args.seed,
        "settings": settings,
        "legacy_parameters": legacy_parameters,
        "vectorized_parameters": vectorized_parameters,
        "parameter_count_preserved": legacy_parameters == vectorized_parameters,
        "maximum_absolute_error": maximum_error,
        "tolerance": tolerance,
        "equivalent": maximum_error <= tolerance and legacy_parameters == vectorized_parameters,
        "shape_trace": traces,
        "optimization": {
            "before": "32 independent Sequential modules executed in a Python loop",
            "after": "two grouped 1x1 convolutions with vectorized GELU and per-group LayerNorm",
            "semantic_contract": "32 independent group-specific dynamic gates; output [B, 768]",
        },
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps(payload, indent=2))
    if not payload["equivalent"]:
        raise RuntimeError("Vectorized grouped gate failed the legacy-equivalence check")


if __name__ == "__main__":
    main()
