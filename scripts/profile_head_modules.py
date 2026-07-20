#!/usr/bin/env python3
"""Profile LiteFusion and supplied-source MKAN heads module by module."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    if parser.parse_known_args()[0].warmup < 30:
        parser.error("warmup must be >=30")
    args = parser.parse_args()
    if args.iterations < 100 or args.rounds < 3:
        parser.error("iterations must be >=100 and rounds >=3")
    return args


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def assert_idle_gpu() -> None:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    ).stdout.strip().splitlines()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    ).stdout.strip().splitlines()
    utilization = float(query[0]) if query else 100.0
    other_pids = [int(value.strip()) for value in apps if value.strip() and int(value.strip()) != os.getpid()]
    if utilization > 10 or other_pids:
        raise RuntimeError(f"GPU busy: utilization={utilization}%, other_compute_pids={other_pids}")


class StudentHead(nn.Module):
    def __init__(self, fusion: nn.Module, gate: nn.Module, classifier: nn.Module):
        super().__init__()
        self.fusion = fusion
        self.gate = gate
        self.classifier = classifier

    def forward(self, vision, text):
        fusion = self.fusion(vision, text)
        gate = self.gate(vision, text, fusion)
        fused = F.normalize(fusion + gate * text + (1.0 - gate) * vision, dim=-1)
        return self.classifier(fused)


def load_heads(root: Path, device: torch.device):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "mkan_refine/reproduction"))
    from kd_litefusion_mkan_teacher.model import LowRankCrossModalFusion, ReliabilityAwareGate
    from model import MKANHead

    student_checkpoint = root / "outputs/formal_multiseed/logits_kd/seed_3407/best_weighted_f1.pt"
    student_payload = load_torch(student_checkpoint)
    train_args = student_payload.get("args", {})
    rank = int(train_args.get("rank", 32))
    dropout = float(train_args.get("dropout", 0.2))
    fusion = LowRankCrossModalFusion(768, rank=rank, dropout=dropout)
    gate = ReliabilityAwareGate(768)
    classifier = nn.Sequential(
        nn.LayerNorm(768), nn.Linear(768, 384), nn.GELU(), nn.Dropout(dropout), nn.Linear(384, 5)
    )
    student = StudentHead(fusion, gate, classifier)
    state = student_payload["student_state_dict"]
    student.load_state_dict(
        {
            key: value
            for key, value in state.items()
            if key.startswith(("fusion.", "gate.", "classifier."))
        },
        strict=True,
    )

    teacher_checkpoint = root / "outputs/server_mkan_kd_formal/checkpoints/ema_seed3407.pth"
    teacher_payload = load_torch(teacher_checkpoint)
    teacher = MKANHead()
    teacher.load_state_dict(teacher_payload["model_state_dict"], strict=True)
    return student.to(device).float().eval(), teacher.to(device).float().eval()


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_rows(student: StudentHead, teacher: nn.Module) -> list[dict]:
    specs = []
    for group, names in {
        "fusion": ("fusion.vision_down", "fusion.text_down", "fusion.interaction_up", "fusion.residual", "fusion.norm"),
        "gate": ("gate.net.0", "gate.net.2", "gate.net.3"),
        "classifier": ("classifier.0", "classifier.1", "classifier.4"),
    }.items():
        for name in names:
            specs.append(("LiteFusion", group, name, student.get_submodule(name)))

    # KAN scale vectors are registered directly on KANLinear, so report them separately.
    specs.extend(
        [
            ("MKAN single", "attention_fusion", "cross_attn.text_pool", teacher.cross_attn.text_pool),
            ("MKAN single", "attention_fusion", "cross_attn.score.base", teacher.cross_attn.score.base),
            ("MKAN single", "attention_fusion", "cross_attn.score.spline", teacher.cross_attn.score.spline),
            ("MKAN single", "gate", "gate.base", teacher.gate.base),
            ("MKAN single", "gate", "gate.spline", teacher.gate.spline),
            ("MKAN single", "classifier", "classifier.0.base", teacher.classifier[0].base),
            ("MKAN single", "classifier", "classifier.0.spline", teacher.classifier[0].spline),
            ("MKAN single", "classifier", "classifier.1", teacher.classifier[1]),
            ("MKAN single", "classifier", "classifier.4.base", teacher.classifier[4].base),
            ("MKAN single", "classifier", "classifier.4.spline", teacher.classifier[4].spline),
        ]
    )
    rows = [
        {"model": model, "group": group, "module": name, "parameters": count_parameters(module), "is_total": False}
        for model, group, name, module in specs
    ]
    direct_parameters = [
        ("MKAN single", "attention_fusion", "cross_attn.score.scale_base", teacher.cross_attn.score.scale_base.numel()),
        ("MKAN single", "attention_fusion", "cross_attn.score.scale_spline", teacher.cross_attn.score.scale_spline.numel()),
        ("MKAN single", "gate", "gate.scale_base", teacher.gate.scale_base.numel()),
        ("MKAN single", "gate", "gate.scale_spline", teacher.gate.scale_spline.numel()),
        ("MKAN single", "classifier", "classifier.0.scale_base", teacher.classifier[0].scale_base.numel()),
        ("MKAN single", "classifier", "classifier.0.scale_spline", teacher.classifier[0].scale_spline.numel()),
        ("MKAN single", "classifier", "classifier.4.scale_base", teacher.classifier[4].scale_base.numel()),
        ("MKAN single", "classifier", "classifier.4.scale_spline", teacher.classifier[4].scale_spline.numel()),
    ]
    rows.extend(
        {"model": model, "group": group, "module": name, "parameters": value, "is_total": False}
        for model, group, name, value in direct_parameters
    )
    for model, module in (("LiteFusion", student), ("MKAN single", teacher)):
        rows.append(
            {"model": model, "group": "all", "module": "HEAD_TOTAL", "parameters": count_parameters(module), "is_total": True}
        )
    return rows


def measure(function: Callable, warmup: int, iterations: int, rounds: int):
    means = []
    with torch.inference_mode():
        for _ in range(rounds):
            for _ in range(warmup):
                function()
            torch.cuda.synchronize()
            samples = []
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                function()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
            means.append(statistics.mean(samples))
    return statistics.mean(means), statistics.stdev(means), means


def flops(function: Callable) -> int:
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode():
        function()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as result:
            function()
        torch.cuda.synchronize()
    return int(sum(int(event.flops or 0) for event in result.key_averages()))


def logical_components(student: StudentHead, teacher: nn.Module, batch: int, device):
    vision = torch.randn(batch, 768, device=device)
    text = torch.randn(batch, 768, device=device)
    vision_tokens = torch.randn(batch, 577, 768, device=device)
    text_tokens = torch.randn(batch, 77, 768, device=device)

    student_low_vision = student.fusion.vision_down(vision)
    student_low_text = student.fusion.text_down(text)
    interaction = student.fusion.interaction_up(student_low_vision * student_low_text)
    residual = student.fusion.residual(torch.cat([vision, text], dim=-1))
    fusion = student.fusion(vision, text)
    evidence = torch.cat([vision, text, torch.abs(vision - text), fusion], dim=-1)
    gate_hidden = student.gate.net[0](evidence)
    gate_middle = student.gate.net[2](student.gate.net[1](gate_hidden))
    gate = student.gate(vision, text, fusion)
    student_fused = F.normalize(fusion + gate * text + (1 - gate) * vision, dim=-1)

    teacher_attention = teacher.cross_attn(text_tokens, vision_tokens)
    teacher_vision = vision + teacher_attention[0]
    teacher_text = text + teacher_attention[1]
    teacher_gate = torch.sigmoid(teacher.gate(torch.cat([teacher_vision, teacher_text], dim=-1)))
    teacher_fused = teacher_vision + teacher_gate * (teacher_text - teacher_vision)

    return {
        "LiteFusion": [
            ("fusion", "vision_projection", lambda: student.fusion.vision_down(vision)),
            ("fusion", "text_projection", lambda: student.fusion.text_down(text)),
            ("fusion", "low_rank_multiply_and_up", lambda: student.fusion.interaction_up(student_low_vision * student_low_text)),
            ("fusion", "residual_projection_1536_to_768", lambda: student.fusion.residual(torch.cat([vision, text], dim=-1))),
            ("fusion", "fusion_norm_dropout_add", lambda: student.fusion.norm(student.fusion.dropout(interaction) + residual)),
            ("fusion", "fusion_full", lambda: student.fusion(vision, text)),
            ("gate", "evidence_concat_absdiff", lambda: torch.cat([vision, text, torch.abs(vision - text), fusion], dim=-1)),
            ("gate", "gate_input_3072_to_768", lambda: student.gate.net[0](evidence)),
            ("gate", "gate_gelu_layernorm", lambda: student.gate.net[2](student.gate.net[1](gate_hidden))),
            ("gate", "gate_output_768_to_768_sigmoid", lambda: student.gate.net[4](student.gate.net[3](gate_middle))),
            ("gate", "gate_full", lambda: student.gate(vision, text, fusion)),
            ("classifier", "classifier_layernorm", lambda: student.classifier[0](student_fused)),
            ("classifier", "classifier_768_to_384", lambda: student.classifier[1](student.classifier[0](student_fused))),
            ("classifier", "classifier_output_to_5", lambda: student.classifier(student_fused)),
            ("all", "HEAD_TOTAL", lambda: student(vision, text)),
        ],
        "MKAN single": [
            ("attention_fusion", "text_pool", lambda: teacher.cross_attn.text_pool(text_tokens)),
            ("attention_fusion", "vision_score", lambda: teacher.cross_attn.score(text_tokens.mean(dim=1, keepdim=True) * vision_tokens)),
            ("attention_fusion", "text_score", lambda: teacher.cross_attn.score(vision_tokens.mean(dim=1, keepdim=True) * text_tokens)),
            ("attention_fusion", "attention_full", lambda: teacher.cross_attn(text_tokens, vision_tokens)),
            ("gate", "gate_1536_to_768", lambda: torch.sigmoid(teacher.gate(torch.cat([teacher_vision, teacher_text], dim=-1)))),
            ("classifier", "classifier_kan_768_to_512", lambda: teacher.classifier[0](teacher_fused)),
            ("classifier", "classifier_norm_activation", lambda: teacher.classifier[2](teacher.classifier[1](teacher.classifier[0](teacher_fused)))),
            ("classifier", "classifier_output_to_5", lambda: teacher.classifier(teacher_fused)),
            ("all", "HEAD_TOTAL", lambda: teacher(text_tokens, vision_tokens, vision, text)["logits"]),
        ],
    }


def shape_trace(model_name: str, module: nn.Module, inputs: tuple) -> list[dict]:
    rows = []
    handles = []
    for name, child in module.named_modules():
        if not name or any(True for _ in child.children()):
            continue
        def hook(_module, args, output, name=name):
            def shape(value):
                if torch.is_tensor(value):
                    return list(value.shape)
                if isinstance(value, (list, tuple)):
                    return [shape(item) for item in value]
                return str(type(value).__name__)
            rows.append({"model": model_name, "module": name, "input_shapes": shape(args), "output_shapes": shape(output)})
        handles.append(child.register_forward_hook(hook))
    with torch.inference_mode():
        module(*inputs)
    for handle in handles:
        handle.remove()
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            for key, value in serial.items():
                if isinstance(value, (list, dict)):
                    serial[key] = json.dumps(value)
            writer.writerow(serial)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    assert_idle_gpu()
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.manual_seed(3407)
    student, teacher = load_heads(root, device)
    params = parameter_rows(student, teacher)
    write_csv(out / "module_parameter_breakdown.csv", params, ["model", "group", "module", "parameters", "is_total"])

    latency_rows, flops_rows, trace_rows = [], [], []
    for batch in sorted(set(args.batch_sizes)):
        components = logical_components(student, teacher, batch, device)
        for model, values in components.items():
            for group, name, function in values:
                mean, std, rounds = measure(function, args.warmup, args.iterations, args.rounds)
                latency_rows.append(
                    {"model": model, "batch_size": batch, "group": group, "module": name,
                     "latency_ms_mean": mean, "latency_ms_std": std, "latency_ms_rounds": rounds}
                )
                if batch == 1:
                    value = flops(function)
                    flops_rows.append(
                        {"model": model, "group": group, "module": name, "flops_per_sample": value,
                         "macs_per_sample_assuming_2_flops_per_mac": value / 2.0,
                         "method": "torch.profiler with_flops=True; unsupported elementwise operations may be omitted"}
                    )
        vision = torch.randn(batch, 768, device=device)
        text = torch.randn(batch, 768, device=device)
        vision_tokens = torch.randn(batch, 577, 768, device=device)
        text_tokens = torch.randn(batch, 77, 768, device=device)
        trace_rows.extend(
            {**row, "batch_size": batch} for row in shape_trace("LiteFusion", student, (vision, text))
        )
        trace_rows.extend(
            {**row, "batch_size": batch}
            for row in shape_trace("MKAN single", teacher, (text_tokens, vision_tokens, vision, text))
        )

    write_csv(
        out / "module_latency_breakdown.csv", latency_rows,
        ["model", "batch_size", "group", "module", "latency_ms_mean", "latency_ms_std", "latency_ms_rounds"],
    )
    write_csv(
        out / "module_flops_breakdown.csv", flops_rows,
        ["model", "group", "module", "flops_per_sample", "macs_per_sample_assuming_2_flops_per_mac", "method"],
    )
    write_csv(
        out / "tensor_shape_trace.csv", trace_rows,
        ["model", "batch_size", "module", "input_shapes", "output_shapes"],
    )
    print(out)


if __name__ == "__main__":
    main()
