#!/usr/bin/env python3
"""Fair V100 efficiency benchmark for MKAN reproduction teachers and LiteFusion.

The benchmark deliberately measures one shared OpenAI CLIP ViT-L/14@336px
encoder plus either one/three MKAN heads or the LiteFusion student head.  The
formal ensemble always executes all three heads, including a member whose
validation-selected weight is zero, so its cost is not understated.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TEACHER_DISPLAY_NAME = "MKAN-Refine supplied-source reproduction teacher"
DEFAULT_CLIP = "/home/lpc/.cache/clip/ViT-L-14-336px.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal fair teacher/student efficiency benchmark")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model-path", default=DEFAULT_CLIP)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.warmup < 30 or args.iterations < 100 or args.rounds < 3:
        parser.error("formal protocol requires warmup>=30, iterations>=100, rounds>=3")
    if sorted(set(args.batch_sizes)) != [1, 8]:
        parser.error("formal protocol requires exactly batch sizes 1 and 8")
    return args


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def count_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def count_trainable(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def valid_inputs(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(batch_size, 3, 336, 336, device=device, dtype=torch.float32)
    tokens = torch.randint(1, 49400, (batch_size, 77), device=device, dtype=torch.long)
    tokens[:, -1] = 49407  # OpenAI CLIP EOT is also the largest token id.
    return images, tokens


def feature_inputs(batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "vision_tokens": torch.randn(batch_size, 577, 768, device=device, dtype=torch.float32),
        "text_tokens": torch.randn(batch_size, 77, 768, device=device, dtype=torch.float32),
        "vision_global": torch.randn(batch_size, 768, device=device, dtype=torch.float32),
        "text_global": torch.randn(batch_size, 768, device=device, dtype=torch.float32),
    }


class MKANRuntime(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        token_encoder_cls,
        head_cls,
        checkpoints: Sequence[Path],
        weights: Sequence[float],
        device: torch.device,
    ) -> None:
        super().__init__()
        if len(checkpoints) != len(weights):
            raise ValueError("checkpoint/weight length mismatch")
        self.clip = clip_model.float().eval()
        self.clip.requires_grad_(False)
        self.encoder = token_encoder_cls(self.clip)
        self.heads = nn.ModuleList()
        for path in checkpoints:
            payload = load_torch(path)
            if payload.get("strict_b_spline_reproduction") is not False:
                raise ValueError(f"invalid strict B-spline claim in {path}")
            head = head_cls().to(device).float().eval()
            head.load_state_dict(payload["model_state_dict"], strict=True)
            self.heads.append(head)
        self.register_buffer("ensemble_weights", torch.tensor(weights, dtype=torch.float32, device=device))
        self.eval()

    def forward_head(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = [head(**features)["logits"] for head in self.heads]
        # All heads are evaluated even when a selected ensemble weight is zero.
        return sum(self.ensemble_weights[index] * value for index, value in enumerate(outputs))

    def forward_fusion_module(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused_values = []
        for head in self.heads:
            vision_enhanced, text_enhanced = head.cross_attn(
                features["text_tokens"], features["vision_tokens"]
            )
            vision_final = features["vision_global"] + vision_enhanced
            text_final = features["text_global"] + text_enhanced
            gate = torch.sigmoid(head.gate(torch.cat([vision_final, text_final], dim=-1)))
            fused_values.append(vision_final + gate * (text_final - vision_final))
        return sum(self.ensemble_weights[index] * value for index, value in enumerate(fused_values))

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        values = self.encoder.encode(images, text_tokens)
        features = dict(zip(("vision_tokens", "text_tokens", "vision_global", "text_global"), values))
        return self.forward_head(features)


class MKANHeadWrapper(nn.Module):
    def __init__(self, runtime: MKANRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    def forward(self, vision_tokens, text_tokens, vision_global, text_global):
        return self.runtime.forward_head(
            {
                "vision_tokens": vision_tokens,
                "text_tokens": text_tokens,
                "vision_global": vision_global,
                "text_global": text_global,
            }
        )


class MKANFusionWrapper(MKANHeadWrapper):
    def forward(self, vision_tokens, text_tokens, vision_global, text_global):
        return self.runtime.forward_fusion_module(
            {
                "vision_tokens": vision_tokens,
                "text_tokens": text_tokens,
                "vision_global": vision_global,
                "text_global": text_global,
            }
        )


class StudentEndToEnd(nn.Module):
    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.student(text_tokens, images)["logits"]


class StudentHeadOnly(nn.Module):
    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(self, vision_global: torch.Tensor, text_global: torch.Tensor) -> torch.Tensor:
        fusion = self.student.fusion(vision_global, text_global)
        gate = self.student.gate(vision_global, text_global, fusion)
        fused = F.normalize(fusion + gate * text_global + (1.0 - gate) * vision_global, dim=-1)
        return self.student.classifier(fused)


class StudentFusionOnly(StudentHeadOnly):
    def forward(self, vision_global: torch.Tensor, text_global: torch.Tensor) -> torch.Tensor:
        fusion = self.student.fusion(vision_global, text_global)
        gate = self.student.gate(vision_global, text_global, fusion)
        return F.normalize(fusion + gate * text_global + (1.0 - gate) * vision_global, dim=-1)


def synchronized_latency(
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    batch_size: int,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict:
    module.eval()
    round_latency, round_throughput = [], []
    with torch.inference_mode():
        for _ in range(rounds):
            for _ in range(warmup):
                module(*inputs)
            torch.cuda.synchronize()
            samples = []
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                module(*inputs)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
            mean_latency = statistics.mean(samples)
            round_latency.append(mean_latency)
            round_throughput.append(batch_size * 1000.0 / mean_latency)
    return {
        "latency_ms_rounds": round_latency,
        "latency_ms_mean": statistics.mean(round_latency),
        "latency_ms_std": statistics.stdev(round_latency),
        "throughput_samples_per_s_rounds": round_throughput,
        "throughput_samples_per_s_mean": statistics.mean(round_throughput),
        "throughput_samples_per_s_std": statistics.stdev(round_throughput),
    }


def peak_memory(module: nn.Module, inputs: Tuple[torch.Tensor, ...], warmup: int) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            module(*inputs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "baseline_bytes": int(baseline),
        "peak_bytes": int(peak),
        "incremental_peak_bytes": int(max(0, peak - baseline)),
    }


def profiler_flops(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> dict:
    from torch.profiler import ProfilerActivity, profile

    module.eval()
    with torch.inference_mode():
        module(*inputs)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
            module(*inputs)
        torch.cuda.synchronize()
    flops = int(sum(int(event.flops or 0) for event in prof.key_averages()))
    if flops <= 0:
        raise RuntimeError("torch.profiler returned zero FLOPs")
    return {
        "flops_per_sample": flops,
        "macs_per_sample_assuming_2_flops_per_mac": flops / 2.0,
        "method": "torch.profiler with_flops=True; unsupported elementwise operations may be omitted",
    }


def benchmark_modes(
    model_key: str,
    modules: Dict[str, nn.Module],
    input_factory,
    device: torch.device,
    args: argparse.Namespace,
) -> List[dict]:
    rows = []
    for batch_size in sorted(set(args.batch_sizes)):
        inputs_by_mode = input_factory(batch_size, device)
        for mode, module in modules.items():
            inputs = inputs_by_mode[mode]
            latency = synchronized_latency(
                module, inputs, batch_size, args.warmup, args.iterations, args.rounds
            )
            memory = peak_memory(module, inputs, args.warmup)
            rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "batch_size": batch_size,
                    **latency,
                    **memory,
                }
            )
    return rows


def student_inputs(batch_size: int, device: torch.device):
    images, tokens = valid_inputs(batch_size, device)
    global_features = (
        torch.randn(batch_size, 768, device=device),
        torch.randn(batch_size, 768, device=device),
    )
    return {
        "end_to_end": (images, tokens),
        "fusion_head_only": global_features,
        "fusion_module_only": global_features,
    }


def teacher_inputs(batch_size: int, device: torch.device):
    images, tokens = valid_inputs(batch_size, device)
    features = feature_inputs(batch_size, device)
    head = (
        features["vision_tokens"],
        features["text_tokens"],
        features["vision_global"],
        features["text_global"],
    )
    return {"end_to_end": (images, tokens), "fusion_head_only": head, "fusion_module_only": head}


def software_metadata(device: torch.device) -> dict:
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "driver_query": driver,
        "precision": "FP32 (CLIP and heads explicitly converted to float32; no autocast)",
        "image_shape": [3, 336, 336],
        "text_length": 77,
        "model_eval": True,
        "inference_context": "torch.inference_mode",
    }


def teacher_breakdown(runtime: MKANRuntime, checkpoints: Sequence[Path], clip_path: Path) -> dict:
    clip_params = count_params(runtime.clip)
    head_params = sum(count_params(head) for head in runtime.heads)
    gate_params = sum(count_params(head.gate) for head in runtime.heads)
    classifier_params = sum(count_params(head.classifier) for head in runtime.heads)
    fusion_core_params = sum(count_params(head.cross_attn) for head in runtime.heads)
    total = clip_params + head_params
    return {
        "total_parameters": total,
        "trainable_parameters": head_params,
        "frozen_parameters": clip_params,
        "clip_parameters": clip_params,
        "fusion_head_parameters": head_params,
        "fusion_core_parameters": fusion_core_params,
        "fusion_plus_gate_parameters": fusion_core_params + gate_params,
        "gate_parameters": gate_params,
        "classifier_parameters": classifier_params,
        "checkpoint_size_bytes": sum(path.stat().st_size for path in checkpoints),
        "clip_checkpoint_size_bytes": clip_path.stat().st_size,
        "deployment_size_bytes": clip_path.stat().st_size + sum(path.stat().st_size for path in checkpoints),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "clip_shared_once": True,
        "heads_executed": len(checkpoints),
    }


def student_breakdown(student: nn.Module, checkpoint: Path, clip_path: Path) -> dict:
    clip_params = count_params(student.clip)
    fusion_params = count_params(student.fusion)
    gate_params = count_params(student.gate)
    classifier_params = count_params(student.classifier)
    trainable = fusion_params + gate_params + classifier_params
    total = clip_params + trainable
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": clip_params,
        "clip_parameters": clip_params,
        "fusion_head_parameters": trainable,
        "fusion_core_parameters": fusion_params,
        "fusion_plus_gate_parameters": fusion_params + gate_params,
        "gate_parameters": gate_params,
        "classifier_parameters": classifier_params,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "clip_checkpoint_size_bytes": clip_path.stat().st_size,
        "deployment_size_bytes": clip_path.stat().st_size + checkpoint.stat().st_size,
        "checkpoint_paths": [str(checkpoint)],
        "clip_shared_once": True,
        "heads_executed": 1,
    }


def find_final_student(root: Path) -> dict | None:
    selection = root / "outputs" / "feature_kd_screening" / "selected_feature_config.yaml"
    manifest_path = root / "outputs" / "feature_kd_screening" / "screening_manifest.json"
    results = root / "outputs" / "feature_kd_screening" / "selected_multiseed_test_results.csv"
    if not (selection.is_file() and manifest_path.is_file() and results.is_file()):
        return None
    import yaml

    selected = yaml.safe_load(selection.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    match = next(
        row
        for row in manifest["trials"]
        if row["condition"] == selected["condition"]
        and math.isclose(float(row["feature_kd_weight"]), float(selected["feature_kd_weight"]))
    )
    checkpoint = Path(match["path"]) / "best_weighted_f1.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {"selection": selected, "checkpoint": str(checkpoint), "performance_csv": str(results)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal efficiency benchmark requires CUDA")
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    clip_path = Path(args.clip_model_path).resolve()
    if not clip_path.is_file():
        raise FileNotFoundError(clip_path)
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "mkan_refine" / "reproduction"))
    import clip
    from kd_litefusion_mkan_teacher.model import KDLiteFusionCLIP
    from kd_litefusion_mkan_teacher.utils import load_checkpoint_state
    from model import MKANHead, OpenAIClipTokenEncoder

    selection_path = root / "outputs" / "server_mkan_kd_formal" / "reports" / "ensemble_selected_weights.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("teacher selection must be validation-only")
    best_seed = int(selection["best_single_seed"])
    checkpoints = [
        root / "outputs" / "server_mkan_kd_formal" / "checkpoints" / f"ema_seed{seed}.pth"
        for seed in (3407, 42, 2024)
    ]
    ensemble_weights = [float(selection["selected_weights"][str(seed)]) for seed in (3407, 42, 2024)]
    single_checkpoint = root / "outputs" / "server_mkan_kd_formal" / "checkpoints" / f"ema_seed{best_seed}.pth"
    student_checkpoint = root / "outputs" / "formal_multiseed" / "logits_kd" / "seed_3407" / "best_weighted_f1.pt"

    raw = {
        "protocol": {
            "teacher_display_name": TEACHER_DISPLAY_NAME,
            "author_original_checkpoint": False,
            "strict_b_spline_kan": False,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "batch_sizes": sorted(set(args.batch_sizes)),
            "clip_model": "OpenAI CLIP ViT-L/14@336px",
            "clip_model_path": str(clip_path),
            "ensemble_executes_all_three_heads": True,
            "end_to_end_scope": "image tensor + token ids -> CLIP -> fusion/gate/classifier -> logits; excludes disk I/O and tokenization",
            "fusion_head_scope": "precomputed CLIP features -> complete fusion/gate/classifier -> logits",
        },
        "environment": software_metadata(device),
        "models": {},
        "measurements": [],
        "flops": [],
        "student_checkpoints": {
            "wo_kd": str(root / "outputs" / "formal_multiseed" / "wo_kd" / "seed_3407" / "best_weighted_f1.pt"),
            "logits_kd": str(student_checkpoint),
        },
        "final_student": find_final_student(root),
    }

    # Teacher best single seed (selected on validation, never on test).
    teacher_specs = [
        ("teacher_single", [single_checkpoint], [1.0]),
        ("teacher_ensemble", checkpoints, ensemble_weights),
    ]
    for model_key, paths, weights in teacher_specs:
        clip_model, _ = clip.load(str(clip_path), device=device, jit=False)
        runtime = MKANRuntime(clip_model, OpenAIClipTokenEncoder, MKANHead, paths, weights, device).to(device)
        modules = {
            "end_to_end": runtime,
            "fusion_head_only": MKANHeadWrapper(runtime).to(device).eval(),
            "fusion_module_only": MKANFusionWrapper(runtime).to(device).eval(),
        }
        raw["models"][model_key] = {
            **teacher_breakdown(runtime, paths, clip_path),
            "best_single_seed": best_seed if model_key == "teacher_single" else None,
            "ensemble_weights": weights if model_key == "teacher_ensemble" else None,
        }
        raw["measurements"].extend(benchmark_modes(model_key, modules, teacher_inputs, device, args))
        inputs = teacher_inputs(1, device)
        for mode, module in modules.items():
            raw["flops"].append(
                {"model_key": model_key, "mode": mode, **profiler_flops(module, inputs[mode])}
            )
        del modules, runtime, clip_model
        gc.collect()
        torch.cuda.empty_cache()

    # One strict runtime measurement is shared by w/o KD, Logits KD, and any
    # selected Feature KD student because training-time KD does not alter the
    # inference graph or load a teacher.
    payload = load_checkpoint_state(str(student_checkpoint), device)
    train_args = payload.get("args", {})
    student = KDLiteFusionCLIP(
        str(clip_path),
        len(payload.get("label_to_id", {})) or 5,
        rank=int(train_args.get("rank", 32)),
        dropout=float(train_args.get("dropout", 0.2)),
        freeze_clip=True,
        device=device,
    ).to(device)
    student.clip.float()
    student.load_student_state_dict(payload["student_state_dict"])
    student.eval()
    modules = {
        "end_to_end": StudentEndToEnd(student).to(device).eval(),
        "fusion_head_only": StudentHeadOnly(student).to(device).eval(),
        "fusion_module_only": StudentFusionOnly(student).to(device).eval(),
    }
    raw["models"]["student_shared"] = student_breakdown(student, student_checkpoint, clip_path)
    raw["measurements"].extend(benchmark_modes("student_shared", modules, student_inputs, device, args))
    inputs = student_inputs(1, device)
    for mode, module in modules.items():
        raw["flops"].append(
            {"model_key": "student_shared", "mode": mode, **profiler_flops(module, inputs[mode])}
        )

    output.mkdir(parents=True, exist_ok=True)
    atomic_json(raw, output / "raw_benchmark.json")
    print(json.dumps({"status": "PASS", "output": str(output / 'raw_benchmark.json')}, indent=2))


if __name__ == "__main__":
    main()
