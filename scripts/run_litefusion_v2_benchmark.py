#!/usr/bin/env python3
"""Formal, fair FP32 benchmark for all LiteFusion-v2 candidates."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kd_litefusion_mkan_teacher.data import read_crisismmd_csv
from kd_litefusion_mkan_teacher.litefusion_v2 import (
    CANDIDATE_NAMES,
    LiteFusionV2Model,
    load_config,
)
from kd_litefusion_mkan_teacher.litefusion_v2.profiling import (
    BenchmarkSettings,
    benchmark_callables_interleaved,
    parameter_breakdown,
    static_head_macs,
    write_profiling_outputs,
)
from kd_litefusion_mkan_teacher.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(PROJECT_ROOT / "configs/litefusion_v2"))
    parser.add_argument(
        "--csv-path",
        default=str(PROJECT_ROOT / "data/clean/task2_clean_consistent.csv"),
    )
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_ROOT / "data/CrisisMMD_v2.0"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/litefusion_v2/benchmark_fp32"),
    )
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATE_NAMES))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--max-start-gpu-util",
        type=int,
        default=10,
        help="Abort before loading a model when initial GPU utilization exceeds this percentage.",
    )
    parser.add_argument(
        "--max-start-gpu-memory-mb",
        type=int,
        default=1024,
        help="Abort when other processes already occupy more than this much GPU memory.",
    )
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def gpu_snapshot() -> Dict[str, object]:
    query = (
        "name,driver_version,memory.total,memory.used,utilization.gpu,"
        "temperature.gpu,power.draw"
    )
    output = subprocess.check_output(
        ("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"),
        text=True,
    ).strip()
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    return {
        "name": values[0],
        "driver_version": values[1],
        "memory_total_mb": float(values[2]),
        "memory_used_mb": float(values[3]),
        "utilization_percent": float(values[4]),
        "temperature_c": float(values[5]),
        "power_draw_w": float(values[6]),
    }


def resolve_image_path(image_root: str, image_path: str) -> Path:
    path = Path(str(image_path))
    return path if path.is_absolute() else Path(image_root) / path


def load_raw_samples(csv_path: str, image_root: str) -> List[Tuple[Image.Image, str]]:
    frame = read_crisismmd_csv(csv_path)
    train = frame.loc[frame["split"].astype(str).str.lower() == "train"].head(8)
    if len(train) != 8:
        raise ValueError("Benchmark requires at least eight train samples")
    samples: List[Tuple[Image.Image, str]] = []
    for row in train.itertuples(index=False):
        path = resolve_image_path(image_root, str(row.image_path))
        try:
            with Image.open(path) as image:
                samples.append((image.convert("RGB").copy(), str(row.text)))
        except Exception as exc:
            raise RuntimeError(f"Failed to load benchmark image {path}: {exc}") from exc
    return samples


def prepare_shared_inputs(
    model: LiteFusionV2Model,
    raw_samples: Sequence[Tuple[Image.Image, str]],
    device: torch.device,
) -> Tuple[Mapping[int, torch.Tensor], Mapping[int, torch.Tensor], Mapping[int, Sequence]]:
    import clip

    images_eight = torch.stack(
        [model.preprocess(image.convert("RGB")) for image, _ in raw_samples]
    ).to(device)
    tokens_eight = clip.tokenize(
        [text for _, text in raw_samples], truncate=True
    ).to(device)
    return (
        {1: images_eight[:1].contiguous(), 8: images_eight},
        {1: tokens_eight[:1].contiguous(), 8: tokens_eight},
        {1: list(raw_samples[:1]), 8: list(raw_samples)},
    )


def add_interleaved_static_metadata(
    rows: Sequence[Mapping[str, object]],
    params_by_candidate: Mapping[str, Mapping[str, int]],
    macs_by_candidate: Mapping[str, Mapping[str, int]],
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for original in rows:
        row = dict(original)
        candidate = str(row["candidate"])
        name = str(row["name"])
        section = name if name in {"fusion", "gate", "classifier"} else "full_head"
        params = params_by_candidate[candidate]
        macs = macs_by_candidate[candidate]
        row.update(
            {
                "head_params": int(params["full_head"]),
                "head_macs": int(macs["full_head"]),
                "component_params": int(params[section]),
                "component_macs": int(macs[section]),
            }
        )
        enriched.append(row)
    return enriched


def benchmark_models_interleaved(
    models: Mapping[str, LiteFusionV2Model],
    images_by_batch: Mapping[int, torch.Tensor],
    tokens_by_batch: Mapping[int, torch.Tensor],
    raw_samples_by_batch: Mapping[int, Sequence],
    device: torch.device,
    settings: BenchmarkSettings,
) -> List[Dict[str, object]]:
    import clip

    rows: List[Dict[str, object]] = []
    first_model = next(iter(models.values()))
    with torch.inference_mode():
        for batch_size in settings.batch_sizes:
            images = images_by_batch[batch_size]
            tokens = tokens_by_batch[batch_size]
            image_feature, text_feature = first_model.encode_clip(images, tokens)
            fusion_features = {
                candidate: model.fusion(image_feature, text_feature)
                for candidate, model in models.items()
            }
            gates = {
                candidate: model.gate(image_feature, text_feature, fusion_features[candidate])
                for candidate, model in models.items()
            }
            final_features = {
                candidate: torch.nn.functional.normalize(
                    fusion_features[candidate]
                    + gates[candidate] * text_feature
                    + (1.0 - gates[candidate]) * image_feature,
                    dim=-1,
                )
                for candidate in models
            }

            def deployment_function(model, samples):
                def call():
                    deployment_images = torch.stack(
                        [model.preprocess(image.convert("RGB")) for image, _ in samples]
                    ).to(device)
                    deployment_tokens = clip.tokenize(
                        [text for _, text in samples], truncate=True
                    ).to(device)
                    return model(deployment_images, deployment_tokens)

                return call

            groups = [
                (
                    "full_model",
                    "head_only",
                    {
                        candidate: (
                            lambda model=model: model.forward_head(image_feature, text_feature)
                        )
                        for candidate, model in models.items()
                    },
                ),
                (
                    "full_model",
                    "gpu_tensor_end_to_end",
                    {
                        candidate: (lambda model=model: model(images, tokens))
                        for candidate, model in models.items()
                    },
                ),
                (
                    "full_model",
                    "deployment_end_to_end",
                    {
                        candidate: deployment_function(
                            model, raw_samples_by_batch[batch_size]
                        )
                        for candidate, model in models.items()
                    },
                ),
                (
                    "clip_only",
                    "component",
                    {
                        candidate: (lambda model=model: model.encode_clip(images, tokens))
                        for candidate, model in models.items()
                    },
                ),
                (
                    "fusion",
                    "component",
                    {
                        candidate: (
                            lambda model=model: model.fusion(image_feature, text_feature)
                        )
                        for candidate, model in models.items()
                    },
                ),
                (
                    "gate",
                    "component",
                    {
                        candidate: (
                            lambda candidate=candidate, model=model: model.gate(
                                image_feature, text_feature, fusion_features[candidate]
                            )
                        )
                        for candidate, model in models.items()
                    },
                ),
                (
                    "classifier",
                    "component",
                    {
                        candidate: (
                            lambda candidate=candidate, model=model: model.classifier(
                                final_features[candidate]
                            )
                        )
                        for candidate, model in models.items()
                    },
                ),
                (
                    "full_head",
                    "component",
                    {
                        candidate: (
                            lambda model=model: model.forward_head(image_feature, text_feature)
                        )
                        for candidate, model in models.items()
                    },
                ),
            ]
            for name, mode, functions in groups:
                rows.extend(
                    benchmark_callables_interleaved(
                        name, mode, batch_size, functions, device, settings
                    )
                )
    return rows


def fairness_report(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    checks = []
    passed = True
    for batch_size in (1, 8):
        clip_rows = [
            row
            for row in rows
            if row["name"] == "clip_only"
            and row["mode"] == "component"
            and int(row["batch_size"]) == batch_size
        ]
        clip_by_candidate = {
            str(row["candidate"]): float(row["mean_ms"]) for row in clip_rows
        }
        if set(clip_by_candidate) != set(CANDIDATE_NAMES):
            raise ValueError(f"Incomplete CLIP-only fairness rows for batch_size={batch_size}")
        clip_values = list(clip_by_candidate.values())
        relative_range = (max(clip_values) - min(clip_values)) / max(
            min(clip_values), 1e-12
        )
        clip_passed = relative_range <= 0.10
        passed = passed and clip_passed
        overhead = {}
        for candidate in CANDIDATE_NAMES:
            matching = [
                row
                for row in rows
                if row["candidate"] == candidate and int(row["batch_size"]) == batch_size
            ]
            clip_ms = next(
                float(row["mean_ms"])
                for row in matching
                if row["name"] == "clip_only" and row["mode"] == "component"
            )
            head_ms = next(
                float(row["mean_ms"])
                for row in matching
                if row["name"] == "full_model" and row["mode"] == "head_only"
            )
            e2e_ms = next(
                float(row["mean_ms"])
                for row in matching
                if row["name"] == "full_model"
                and row["mode"] == "gpu_tensor_end_to_end"
            )
            deployment_ms = next(
                float(row["mean_ms"])
                for row in matching
                if row["name"] == "full_model"
                and row["mode"] == "deployment_end_to_end"
            )
            overhead[candidate] = {
                "clip_only_mean_ms": clip_ms,
                "full_head_mean_ms": head_ms,
                "gpu_tensor_end_to_end_mean_ms": e2e_ms,
                "gpu_tensor_fixed_overhead_ms": e2e_ms - clip_ms - head_ms,
                "deployment_end_to_end_mean_ms": deployment_ms,
                "deployment_fixed_overhead_ms": deployment_ms - clip_ms - head_ms,
            }
        checks.append(
            {
                "batch_size": batch_size,
                "clip_only_mean_ms_by_candidate": clip_by_candidate,
                "clip_only_relative_range": relative_range,
                "clip_only_within_10_percent": clip_passed,
                "latency_decomposition": overhead,
            }
        )
    return {"passed": passed, "checks": checks}


def main() -> None:
    args = parse_args()
    settings = BenchmarkSettings(
        batch_sizes=(1, 8),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    settings.validate()
    if tuple(args.candidates) != tuple(CANDIDATE_NAMES):
        raise ValueError(f"Formal benchmark requires all candidates in order: {CANDIDATE_NAMES}")
    if args.seed != 3407:
        raise ValueError("Formal benchmark is fixed to seed=3407")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal benchmark requires CUDA")

    occupancy_before = gpu_snapshot()
    if occupancy_before["utilization_percent"] > args.max_start_gpu_util:
        raise RuntimeError(
            "GPU is already heavily used: "
            f"{occupancy_before['utilization_percent']:.0f}% > {args.max_start_gpu_util}%"
        )
    if occupancy_before["memory_used_mb"] > args.max_start_gpu_memory_mb:
        raise RuntimeError(
            "GPU memory is already occupied: "
            f"{occupancy_before['memory_used_mb']:.0f} MiB > "
            f"{args.max_start_gpu_memory_mb} MiB"
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty benchmark directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    (output_dir / "benchmark_command.txt").write_text(
        f"cwd={Path.cwd()}\ncommand={command}\n", encoding="utf-8"
    )

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    raw_samples = load_raw_samples(args.csv_path, args.image_root)
    models: Dict[str, LiteFusionV2Model] = {}
    params_by_candidate: Dict[str, Dict[str, int]] = {}
    macs_by_candidate: Dict[str, Dict[str, int]] = {}
    shared_clip = None
    shared_preprocess = None
    for candidate in args.candidates:
        config = load_config(str(Path(args.config_dir) / f"{candidate}.yaml"))
        if config.name != candidate or config.interaction_rank != 32:
            raise ValueError(f"Invalid formal candidate config: {config.to_dict()}")
        model = LiteFusionV2Model(
            config, device=device, load_clip=(shared_clip is None)
        ).to(device).eval()
        if shared_clip is None:
            shared_clip = model.clip
            shared_preprocess = model.preprocess
        else:
            model.clip = shared_clip
            model.preprocess = shared_preprocess
        if model.clip.training:
            raise AssertionError("Frozen CLIP must remain in eval mode")
        models[candidate] = model
        params_by_candidate[candidate] = parameter_breakdown(model)
        macs_by_candidate[candidate] = static_head_macs(
            model, batch_size=1, device=device
        )
    images_by_batch, tokens_by_batch, raw_by_batch = prepare_shared_inputs(
        next(iter(models.values())), raw_samples, device
    )
    rows = benchmark_models_interleaved(
        models,
        images_by_batch,
        tokens_by_batch,
        raw_by_batch,
        device,
        settings,
    )
    all_rows = add_interleaved_static_metadata(
        rows, params_by_candidate, macs_by_candidate
    )

    fairness = fairness_report(all_rows)
    environment = {
        "gpu": occupancy_before["name"],
        "gpu_snapshot_before": occupancy_before,
        "gpu_snapshot_after": gpu_snapshot(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "pytorch": torch.__version__,
        "dtype": "fp32",
        "batch_sizes": [1, 8],
        "warmup": settings.warmup,
        "iterations": settings.iterations,
        "repeats": settings.repeats,
        "seed": args.seed,
        "num_workers": 0,
        "git_commit": git_commit(),
        "csv_path": str(Path(args.csv_path).resolve()),
        "image_root": str(Path(args.image_root).resolve()),
        "candidates": list(args.candidates),
        "timing_order": "candidate_round_robin_interleaved",
        "shared_frozen_clip_instance": True,
        "fairness": fairness,
    }
    write_profiling_outputs(
        all_rows,
        str(output_dir / "benchmark_results.json"),
        str(output_dir / "benchmark_results.csv"),
        metadata=environment,
    )
    with (output_dir / "benchmark_environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, ensure_ascii=False)
    print(pd.DataFrame(all_rows).to_string(index=False))
    print(json.dumps(fairness, indent=2, ensure_ascii=False))
    if not fairness["passed"]:
        raise RuntimeError("CLIP-only latency differs by more than 10%; benchmark is not fair")


if __name__ == "__main__":
    main()
