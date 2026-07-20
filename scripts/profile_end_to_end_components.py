#!/usr/bin/env python3
"""Diagnose MKAN/LiteFusion end-to-end latency with isolated model processes.

Each model is measured in a fresh Python process so CUDA allocations, module
references, and clock history from a previous model cannot leak into the next
measurement. The script records both the current native path and a canonical
shared token-level CLIP path.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MODEL_KEYS = ("teacher_single", "teacher_ensemble", "student_shared")
MODEL_LABELS = {
    "teacher_single": "MKAN-Refine supplied-source reproduction teacher (best single)",
    "teacher_ensemble": "MKAN-Refine supplied-source reproduction teacher (formal 3-model ensemble)",
    "student_shared": "LiteFusion-CLIP student",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model-path", default="/home/lpc/.cache/clip/ViT-L-14-336px.pt")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--worker", choices=MODEL_KEYS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()
    if args.warmup < 30 or args.iterations < 100 or args.rounds < 3:
        parser.error("diagnosis requires warmup>=30, iterations>=100, rounds>=3")
    if sorted(set(args.batch_sizes)) != [1, 8]:
        parser.error("diagnosis requires batch sizes 1 and 8")
    if bool(args.worker) != bool(args.worker_output):
        parser.error("--worker and --worker-output must be provided together")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nvidia_snapshot() -> dict:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {"gpu_query": query.stdout.strip(), "compute_apps": apps.stdout.strip()}


def assert_gpu_idle(allow_busy: bool) -> None:
    if allow_busy:
        return
    snapshot = nvidia_snapshot()
    line = snapshot["gpu_query"].splitlines()[0]
    fields = [item.strip() for item in line.split(",")]
    utilization = float(fields[2]) if len(fields) > 2 else 100.0
    other_pids = []
    for app in snapshot["compute_apps"].splitlines():
        if not app.strip():
            continue
        try:
            pid = int(app.split(",", 1)[0].strip())
        except ValueError:
            continue
        if pid != os.getpid():
            other_pids.append(pid)
    if utilization > 10 or other_pids:
        raise RuntimeError(
            "GPU is not idle enough for diagnosis: "
            f"utilization={utilization:.1f}%, other_compute_pids={other_pids}, snapshot={snapshot}. "
            "Do not use --allow-busy-gpu for formal numbers."
        )


def measure_cuda(
    function: Callable[[], object], warmup: int, iterations: int, rounds: int
) -> tuple[float, float, list[float]]:
    round_means = []
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
            round_means.append(statistics.mean(samples))
    return statistics.mean(round_means), statistics.stdev(round_means), round_means


def measure_cpu(function: Callable[[], object], warmup: int, iterations: int, rounds: int):
    round_means = []
    for _ in range(rounds):
        for _ in range(warmup):
            function()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            function()
            samples.append((time.perf_counter() - start) * 1000.0)
        round_means.append(statistics.mean(samples))
    return statistics.mean(round_means), statistics.stdev(round_means), round_means


def append_timing(rows: list[dict], model_key: str, batch: int, path: str, component: str, values, executed=True):
    mean, std, rounds = values if executed else (0.0, 0.0, [])
    rows.append(
        {
            "model_key": model_key,
            "model": MODEL_LABELS[model_key],
            "batch_size": batch,
            "path": path,
            "component": component,
            "executed": bool(executed),
            "latency_ms_mean": mean,
            "latency_ms_std": std,
            "latency_ms_rounds": rounds,
        }
    )


def manual_image_encode(clip_model, images):
    visual = clip_model.visual
    dtype = clip_model.dtype
    x = visual.conv1(images.to(dtype=dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([class_token, x], dim=1)
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    tokens = x @ visual.proj if visual.proj is not None else x
    return tokens, tokens[:, 0]


def manual_text_encode(clip_model, text_tokens):
    dtype = clip_model.dtype
    x = clip_model.token_embedding(text_tokens).to(dtype)
    x = x + clip_model.positional_embedding.to(dtype)
    x = clip_model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = clip_model.ln_final(x).to(dtype)
    tokens = x @ clip_model.text_projection
    positions = text_tokens.argmax(dim=-1)
    global_feature = tokens[torch.arange(tokens.shape[0], device=tokens.device), positions]
    return tokens, global_feature


def call_count_audit(clip_model, function: Callable[[], object]) -> dict:
    counts = {"vision_conv1": 0, "vision_transformer": 0, "text_embedding": 0, "text_transformer": 0}
    handles = []
    for key, module in (
        ("vision_conv1", clip_model.visual.conv1),
        ("vision_transformer", clip_model.visual.transformer),
        ("text_embedding", clip_model.token_embedding),
        ("text_transformer", clip_model.transformer),
    ):
        handles.append(module.register_forward_pre_hook(lambda _m, _i, key=key: counts.__setitem__(key, counts[key] + 1)))
    with torch.inference_mode():
        function()
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    return counts


def load_runtime(root: Path, clip_path: Path, model_key: str, device: torch.device):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "mkan_refine" / "reproduction"))
    import clip
    from efficiency import MKANRuntime, StudentEndToEnd, load_torch
    from kd_litefusion_mkan_teacher.model import KDLiteFusionCLIP
    from kd_litefusion_mkan_teacher.utils import load_checkpoint_state
    from model import MKANHead, OpenAIClipTokenEncoder

    if model_key.startswith("teacher"):
        selection = json.loads(
            (root / "outputs/server_mkan_kd_formal/reports/ensemble_selected_weights.json").read_text(
                encoding="utf-8"
            )
        )
        best_seed = int(selection["best_single_seed"])
        if model_key == "teacher_single":
            checkpoints = [
                root / "outputs/server_mkan_kd_formal/checkpoints" / f"ema_seed{best_seed}.pth"
            ]
            weights = [1.0]
        else:
            seeds = (3407, 42, 2024)
            checkpoints = [
                root / "outputs/server_mkan_kd_formal/checkpoints" / f"ema_seed{seed}.pth"
                for seed in seeds
            ]
            weights = [float(selection["selected_weights"][str(seed)]) for seed in seeds]
        clip_model, preprocess = clip.load(str(clip_path), device=device, jit=False)
        runtime = MKANRuntime(
            clip_model, OpenAIClipTokenEncoder, MKANHead, checkpoints, weights, device
        ).to(device).float().eval()
        return runtime, runtime.clip, preprocess, runtime

    checkpoint = root / "outputs/formal_multiseed/logits_kd/seed_3407/best_weighted_f1.pt"
    payload = load_checkpoint_state(str(checkpoint), device)
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
    return student, student.clip, student.preprocess, StudentEndToEnd(student).to(device).eval()


def head_components(model_key: str, runtime, vision_tokens, text_tokens, vision_global, text_global):
    if model_key.startswith("teacher"):
        def fusion():
            values = []
            for head in runtime.heads:
                vi, te = head.cross_attn(text_tokens.float(), vision_tokens.float())
                values.append((vision_global.float() + vi, text_global.float() + te))
            return values

        fusion_values = fusion()

        def gate():
            return [
                torch.sigmoid(head.gate(torch.cat([vi, te], dim=-1)))
                for head, (vi, te) in zip(runtime.heads, fusion_values)
            ]

        gate_values = gate()

        def classifier():
            logits = []
            for head, (vi, te), gate_value in zip(runtime.heads, fusion_values, gate_values):
                logits.append(head.classifier(vi + gate_value * (te - vi)))
            return sum(runtime.ensemble_weights[i] * value for i, value in enumerate(logits))

        def head_full():
            features = {
                "vision_tokens": vision_tokens,
                "text_tokens": text_tokens,
                "vision_global": vision_global,
                "text_global": text_global,
            }
            return runtime.forward_head(features)

        return fusion, gate, classifier, head_full, False

    student = runtime
    vision_global = vision_global.float()
    text_global = text_global.float()

    def fusion():
        return student.fusion(vision_global, text_global)

    fusion_value = fusion()

    def gate():
        return student.gate(vision_global, text_global, fusion_value)

    gate_value = gate()

    def classifier():
        fused = F.normalize(
            fusion_value + gate_value * text_global + (1.0 - gate_value) * vision_global,
            dim=-1,
        )
        return student.classifier(fused)

    def head_full():
        local_fusion = student.fusion(vision_global, text_global)
        local_gate = student.gate(vision_global, text_global, local_fusion)
        fused = F.normalize(
            local_fusion + local_gate * text_global + (1.0 - local_gate) * vision_global,
            dim=-1,
        )
        return student.classifier(fused)

    return fusion, gate, classifier, head_full, True


def canonical_head_forward(model_key: str, runtime, vision_tokens, text_tokens, vision_global, text_global):
    """Run exactly one head path without component-profiler precomputation."""
    if model_key.startswith("teacher"):
        return runtime.forward_head(
            {
                "vision_tokens": vision_tokens,
                "text_tokens": text_tokens,
                "vision_global": vision_global,
                "text_global": text_global,
            }
        )
    vision_global = F.normalize(vision_global.float(), dim=-1)
    text_global = F.normalize(text_global.float(), dim=-1)
    fusion = runtime.fusion(vision_global, text_global)
    gate = runtime.gate(vision_global, text_global, fusion)
    fused = F.normalize(
        fusion + gate * text_global + (1.0 - gate) * vision_global,
        dim=-1,
    )
    return runtime.classifier(fused)


def worker(args: argparse.Namespace) -> None:
    assert_gpu_idle(args.allow_busy_gpu)
    root = args.project_root.resolve()
    clip_path = Path(args.clip_model_path).resolve()
    device = torch.device("cuda:0")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    runtime, clip_model, preprocess, native_module = load_runtime(root, clip_path, args.worker, device)
    rows: list[dict] = []
    preprocessing_rows: list[dict] = []
    audits: dict = {"model_key": args.worker, "snapshots": {"start": nvidia_snapshot()}, "batches": {}}

    for batch in sorted(set(args.batch_sizes)):
        cpu_images = torch.randn(batch, 3, 336, 336, dtype=torch.float32)
        cpu_tokens = torch.randint(1, 49400, (batch, 77), dtype=torch.long)
        cpu_tokens[:, -1] = 49407
        images = cpu_images.to(device)
        tokens = cpu_tokens.to(device)
        raw_images = [
            Image.fromarray(np.random.default_rng(args.seed + index).integers(0, 256, (420, 500, 3), dtype=np.uint8))
            for index in range(batch)
        ]
        raw_texts = [f"synthetic CrisisMMD diagnosis sample {index}" for index in range(batch)]

        def cpu_prepare():
            import clip
            return torch.stack([preprocess(image) for image in raw_images]), clip.tokenize(raw_texts, truncate=True)

        prepared_images, prepared_tokens = cpu_prepare()
        prepared_images = prepared_images.pin_memory()
        prepared_tokens = prepared_tokens.pin_memory()

        def host_to_device():
            return (
                prepared_images.to(device, non_blocking=True),
                prepared_tokens.to(device, non_blocking=True),
            )

        cpu_values = measure_cpu(cpu_prepare, args.warmup, args.iterations, args.rounds)
        h2d_values = measure_cuda(host_to_device, args.warmup, args.iterations, args.rounds)
        preprocessing_rows.extend(
            [
                {
                    "model_key": args.worker,
                    "model": MODEL_LABELS[args.worker],
                    "batch_size": batch,
                    "component": "preprocess_and_tokenize_cpu",
                    "latency_ms_mean": cpu_values[0],
                    "latency_ms_std": cpu_values[1],
                    "latency_ms_rounds": cpu_values[2],
                },
                {
                    "model_key": args.worker,
                    "model": MODEL_LABELS[args.worker],
                    "batch_size": batch,
                    "component": "cpu_to_gpu_non_blocking",
                    "latency_ms_mean": h2d_values[0],
                    "latency_ms_std": h2d_values[1],
                    "latency_ms_rounds": h2d_values[2],
                },
            ]
        )

        native = lambda: native_module(images, tokens)
        append_timing(
            rows,
            args.worker,
            batch,
            "current_native_gpu_tensor",
            "end_to_end",
            measure_cuda(native, args.warmup, args.iterations, args.rounds),
        )

        image_encode = lambda: manual_image_encode(clip_model, images)
        text_encode = lambda: manual_text_encode(clip_model, tokens)
        vision_tokens, vision_global = image_encode()
        text_token_features, text_global = text_encode()

        append_timing(
            rows, args.worker, batch, "canonical_shared_token_encoder", "clip_image_encode",
            measure_cuda(image_encode, args.warmup, args.iterations, args.rounds),
        )
        append_timing(
            rows, args.worker, batch, "canonical_shared_token_encoder", "clip_text_encode",
            measure_cuda(text_encode, args.warmup, args.iterations, args.rounds),
        )

        dtype_conversion = lambda: (
            vision_tokens.float(), text_token_features.float(), vision_global.float(), text_global.float()
        )
        append_timing(
            rows, args.worker, batch, "canonical_shared_token_encoder", "feature_dtype_conversion",
            measure_cuda(dtype_conversion, args.warmup, args.iterations, args.rounds),
        )
        normalize_executed = args.worker == "student_shared"
        normalize = lambda: (F.normalize(vision_global.float(), dim=-1), F.normalize(text_global.float(), dim=-1))
        append_timing(
            rows, args.worker, batch, "canonical_shared_token_encoder", "input_l2_normalize",
            measure_cuda(normalize, args.warmup, args.iterations, args.rounds) if normalize_executed else None,
            executed=normalize_executed,
        )
        if normalize_executed:
            vision_global, text_global = normalize()

        fusion, gate, classifier, head_full, _ = head_components(
            args.worker, runtime, vision_tokens, text_token_features, vision_global, text_global
        )
        for component, function in (
            ("fusion", fusion),
            ("gate", gate),
            ("classifier_and_logits", classifier),
            ("fusion_gate_classifier", head_full),
        ):
            append_timing(
                rows, args.worker, batch, "canonical_shared_token_encoder", component,
                measure_cuda(function, args.warmup, args.iterations, args.rounds),
            )

        def canonical_total():
            vi_tokens, vi_global = manual_image_encode(clip_model, images)
            te_tokens, te_global = manual_text_encode(clip_model, tokens)
            return canonical_head_forward(
                args.worker, runtime, vi_tokens, te_tokens, vi_global, te_global
            )

        append_timing(
            rows, args.worker, batch, "canonical_shared_token_encoder", "end_to_end",
            measure_cuda(canonical_total, args.warmup, args.iterations, args.rounds),
        )

        def deployment_total():
            local_images, local_tokens = cpu_prepare()
            local_images = local_images.to(device)
            local_tokens = local_tokens.to(device)
            return native_module(local_images, local_tokens)

        append_timing(
            rows, args.worker, batch, "deployment_raw_image_text", "end_to_end",
            measure_cuda(deployment_total, args.warmup, args.iterations, args.rounds),
        )

        native_counts = call_count_audit(clip_model, native)
        canonical_counts = call_count_audit(clip_model, canonical_total)
        with torch.inference_mode():
            native_output = native()
            if isinstance(native_output, dict):
                native_output = native_output["logits"]
        audits["batches"][str(batch)] = {
            "native_clip_call_counts_per_forward": native_counts,
            "canonical_clip_call_counts_per_forward": canonical_counts,
            "input_device": str(images.device),
            "image_dtype": str(images.dtype),
            "token_dtype": str(tokens.dtype),
            "clip_dtype": str(clip_model.dtype),
            "logits_device": str(native_output.device),
            "logits_dtype": str(native_output.dtype),
            "logits_shape": list(native_output.shape),
        }

    audits["snapshots"]["end"] = nvidia_snapshot()
    payload = {
        "model_key": args.worker,
        "component_rows": rows,
        "preprocessing_rows": preprocessing_rows,
        "audit": audits,
    }
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            if "latency_ms_rounds" in serial:
                serial["latency_ms_rounds"] = json.dumps(serial["latency_ms_rounds"])
            writer.writerow(serial)


def orchestrator(args: argparse.Namespace) -> None:
    assert_gpu_idle(args.allow_busy_gpu)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tmp_root = Path("/dev/shm/lpc_kdclip_tmp/efficiency_diagnosis")
    tmp_root.mkdir(parents=True, exist_ok=True)
    payloads = []
    for model_key in MODEL_KEYS:
        worker_output = tmp_root / f"{model_key}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--project-root", str(args.project_root.resolve()),
            "--output-dir", str(output),
            "--clip-model-path", str(Path(args.clip_model_path).resolve()),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--rounds", str(args.rounds),
            "--batch-sizes", *[str(value) for value in args.batch_sizes],
            "--seed", str(args.seed),
            "--worker", model_key,
            "--worker-output", str(worker_output),
        ]
        if args.allow_busy_gpu:
            command.append("--allow-busy-gpu")
        subprocess.run(command, check=True)
        payloads.append(json.loads(worker_output.read_text(encoding="utf-8")))

    component_rows = [row for payload in payloads for row in payload["component_rows"]]
    preprocessing_rows = [row for payload in payloads for row in payload["preprocessing_rows"]]
    write_csv(
        output / "component_latency.csv",
        component_rows,
        [
            "model_key", "model", "batch_size", "path", "component", "executed",
            "latency_ms_mean", "latency_ms_std", "latency_ms_rounds",
        ],
    )
    write_csv(
        output / "preprocessing_latency.csv",
        preprocessing_rows,
        [
            "model_key", "model", "batch_size", "component", "latency_ms_mean",
            "latency_ms_std", "latency_ms_rounds",
        ],
    )
    clip_rows = [
        row for row in component_rows
        if row["path"] == "canonical_shared_token_encoder"
        and row["component"] in ("clip_image_encode", "clip_text_encode")
    ]
    write_csv(
        output / "clip_latency_comparison.csv",
        clip_rows,
        [
            "model_key", "model", "batch_size", "path", "component", "executed",
            "latency_ms_mean", "latency_ms_std", "latency_ms_rounds",
        ],
    )
    audit = {
        "protocol": {
            "process_isolation": True,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "batch_sizes": sorted(set(args.batch_sizes)),
            "gpu_tensor_boundary": "GPU images + token ids -> CLIP -> head -> logits",
            "deployment_boundary": "in-memory raw PIL images + raw text -> preprocess/tokenize -> GPU -> CLIP -> head -> logits",
        },
        "models": {payload["model_key"]: payload["audit"] for payload in payloads},
    }
    (output / "dtype_and_device_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.worker:
        worker(args)
    else:
        orchestrator(args)


if __name__ == "__main__":
    main()
