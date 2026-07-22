import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from .model import LiteFusionV2Model


@dataclass(frozen=True)
class BenchmarkSettings:
    batch_sizes: Tuple[int, ...] = (1, 8)
    warmup: int = 50
    iterations: int = 200
    repeats: int = 5

    def validate(self) -> None:
        if tuple(self.batch_sizes) != (1, 8):
            raise ValueError("Formal benchmark requires batch_sizes=(1, 8)")
        if self.warmup < 50 or self.iterations < 200 or self.repeats < 5:
            raise ValueError("Formal benchmark requires warmup>=50, iterations>=200, repeats>=5")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    name: str,
    mode: str,
    batch_size: int,
    function: Callable[[], Any],
    device: torch.device,
    settings: BenchmarkSettings,
) -> Dict[str, Any]:
    settings.validate()
    for _ in range(settings.warmup):
        function()
        synchronize(device)

    samples_ms: List[float] = []
    peak_memory = 0
    for _ in range(settings.repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(settings.iterations):
            synchronize(device)
            started = time.perf_counter_ns()
            function()
            synchronize(device)
            samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
    ordered = sorted(samples_ms)
    p50 = ordered[int(0.50 * (len(ordered) - 1))]
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return {
        "name": name,
        "mode": mode,
        "batch_size": int(batch_size),
        "dtype": "fp32",
        "warmup": settings.warmup,
        "iterations": settings.iterations,
        "repeats": settings.repeats,
        "mean_ms": statistics.fmean(samples_ms),
        "std_ms": statistics.pstdev(samples_ms),
        "p50_ms": p50,
        "p95_ms": p95,
        "peak_gpu_memory_bytes": peak_memory,
    }


def benchmark_callables_interleaved(
    name: str,
    mode: str,
    batch_size: int,
    functions: Mapping[str, Callable[[], Any]],
    device: torch.device,
    settings: BenchmarkSettings,
) -> List[Dict[str, Any]]:
    """Benchmark equivalent candidates round-robin to reduce changing-load bias."""
    settings.validate()
    if not functions:
        raise ValueError("At least one benchmark callable is required")
    for _ in range(settings.warmup):
        for function in functions.values():
            function()
            synchronize(device)

    samples_by_candidate: Dict[str, List[float]] = {
        candidate: [] for candidate in functions
    }
    peak_by_candidate = {candidate: 0 for candidate in functions}
    for _ in range(settings.repeats):
        for _ in range(settings.iterations):
            for candidate, function in functions.items():
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                synchronize(device)
                started = time.perf_counter_ns()
                function()
                synchronize(device)
                samples_by_candidate[candidate].append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )
                if device.type == "cuda":
                    peak_by_candidate[candidate] = max(
                        peak_by_candidate[candidate],
                        int(torch.cuda.max_memory_allocated(device)),
                    )

    rows: List[Dict[str, Any]] = []
    for candidate, samples_ms in samples_by_candidate.items():
        ordered = sorted(samples_ms)
        rows.append(
            {
                "name": name,
                "mode": mode,
                "batch_size": int(batch_size),
                "dtype": "fp32",
                "warmup": settings.warmup,
                "iterations": settings.iterations,
                "repeats": settings.repeats,
                "mean_ms": statistics.fmean(samples_ms),
                "std_ms": statistics.pstdev(samples_ms),
                "p50_ms": ordered[int(0.50 * (len(ordered) - 1))],
                "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
                "peak_gpu_memory_bytes": peak_by_candidate[candidate],
                "candidate": candidate,
            }
        )
    return rows


def count_parameters(module_or_parameters: Any) -> int:
    parameters = module_or_parameters.parameters() if isinstance(module_or_parameters, nn.Module) else module_or_parameters
    return sum(parameter.numel() for parameter in parameters)


def parameter_breakdown(model: LiteFusionV2Model) -> Dict[str, int]:
    breakdown = {
        "fusion": count_parameters(model.fusion),
        "gate": count_parameters(model.gate),
        "classifier": count_parameters(model.classifier),
    }
    breakdown["full_head"] = sum(breakdown.values())
    return breakdown


def static_head_macs(
    model: LiteFusionV2Model,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, int]:
    totals = {"fusion": 0, "gate": 0, "classifier": 0}
    handles = []

    def register(module: nn.Module, section: str) -> None:
        def linear_hook(layer: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            input_tensor = inputs[0]
            vectors = input_tensor.numel() // input_tensor.shape[-1]
            totals[section] += int(vectors * layer.in_features * layer.out_features)

        def conv1d_hook(layer: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            kernel = int(layer.kernel_size[0])
            inputs_per_group = int(layer.in_channels // layer.groups)
            totals[section] += int(output.numel() * inputs_per_group * kernel)

        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                handles.append(layer.register_forward_hook(linear_hook))
            elif isinstance(layer, nn.Conv1d):
                handles.append(layer.register_forward_hook(conv1d_hook))

    register(model.fusion, "fusion")
    register(model.gate, "gate")
    register(model.classifier, "classifier")
    image_feature = torch.randn(batch_size, model.config.feature_dim, device=device)
    text_feature = torch.randn(batch_size, model.config.feature_dim, device=device)
    model.eval()
    with torch.inference_mode():
        model.forward_head(image_feature, text_feature)
    for handle in handles:
        handle.remove()
    totals["full_head"] = sum(totals.values())
    return totals


def profile_components(
    model: LiteFusionV2Model,
    images: torch.Tensor,
    tokens: torch.Tensor,
    image_feature: torch.Tensor,
    text_feature: torch.Tensor,
    device: torch.device,
    settings: BenchmarkSettings,
) -> List[Dict[str, Any]]:
    model.eval()
    batch_size = int(image_feature.shape[0])
    with torch.inference_mode():
        fusion_feature = model.fusion(image_feature, text_feature)
        gate = model.gate(image_feature, text_feature, fusion_feature)
        final_feature = torch.nn.functional.normalize(
            fusion_feature + gate * text_feature + (1.0 - gate) * image_feature,
            dim=-1,
        )
    callables = {
        "clip_only": lambda: model.encode_clip(images, tokens),
        "fusion": lambda: model.fusion(image_feature, text_feature),
        "gate": lambda: model.gate(image_feature, text_feature, fusion_feature),
        "classifier": lambda: model.classifier(final_feature),
        "full_head": lambda: model.forward_head(image_feature, text_feature),
    }
    rows = []
    with torch.inference_mode():
        for name, function in callables.items():
            rows.append(
                benchmark_callable(name, "component", batch_size, function, device, settings)
            )
    return rows


def benchmark_modes(
    model: LiteFusionV2Model,
    images_by_batch: Mapping[int, torch.Tensor],
    tokens_by_batch: Mapping[int, torch.Tensor],
    raw_samples_by_batch: Mapping[int, Sequence[Tuple[Any, str]]],
    device: torch.device,
    settings: BenchmarkSettings = BenchmarkSettings(),
) -> List[Dict[str, Any]]:
    import clip

    settings.validate()
    model.eval()
    rows: List[Dict[str, Any]] = []
    for batch_size in settings.batch_sizes:
        images = images_by_batch[batch_size]
        tokens = tokens_by_batch[batch_size]
        with torch.inference_mode():
            image_feature, text_feature = model.encode_clip(images, tokens)

        def deployment_call():
            samples = raw_samples_by_batch[batch_size]
            deployment_images = torch.stack([model.preprocess(image.convert("RGB")) for image, _ in samples]).to(device)
            deployment_tokens = clip.tokenize([text for _, text in samples], truncate=True).to(device)
            return model(deployment_images, deployment_tokens)

        mode_functions = {
            "head_only": lambda: model.forward_head(image_feature, text_feature),
            "gpu_tensor_end_to_end": lambda: model(images, tokens),
            "deployment_end_to_end": deployment_call,
        }
        with torch.inference_mode():
            for mode, function in mode_functions.items():
                rows.append(
                    benchmark_callable("full_model", mode, batch_size, function, device, settings)
                )
        rows.extend(
            profile_components(
                model,
                images,
                tokens,
                image_feature,
                text_feature,
                device,
                settings,
            )
        )
    return rows


def write_profiling_outputs(
    rows: Iterable[Mapping[str, Any]],
    json_path: str,
    csv_path: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    rows = [dict(row) for row in rows]
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": dict(metadata or {}), "results": rows}, handle, indent=2, ensure_ascii=False)
    fieldnames = sorted({key for row in rows for key in row})
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
