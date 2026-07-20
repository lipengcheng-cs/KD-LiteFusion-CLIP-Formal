#!/usr/bin/env python3
"""Summarize static and measured causes of the preliminary latency anomaly."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


PATTERNS = {
    ".cpu()": r"\.cpu\s*\(",
    ".numpy()": r"\.numpy\s*\(",
    ".item()": r"\.item\s*\(",
    ".clone()": r"\.clone\s*\(",
    ".to(...)": r"\.to\s*\(",
    "F.normalize": r"F\.normalize\s*\(",
    "tokenize": r"\btokenize\s*\(",
    "preprocess": r"\bpreprocess\s*\(",
    "DataLoader": r"\bDataLoader\b",
    "PIL image": r"\bImage\.",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    parser.add_argument("--old-raw", type=Path, required=True)
    return parser.parse_args()


def scan_source(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    result = {}
    for label, pattern in PATTERNS.items():
        matches = [index for index, line in enumerate(lines, start=1) if re.search(pattern, line)]
        result[label] = matches
    return result


def fmt(value: float) -> str:
    return f"{value:.3f}"


def old_measurement_table(raw: dict) -> pd.DataFrame:
    rows = []
    for row in raw["measurements"]:
        rows.append(
            {
                "model_key": row["model_key"],
                "mode": row["mode"],
                "batch_size": int(row["batch_size"]),
                "latency": float(row["latency_ms_mean"]),
                "std": float(row["latency_ms_std"]),
                "baseline_mib": float(row["baseline_bytes"]) / 2**20,
                "peak_mib": float(row["peak_bytes"]) / 2**20,
                "incremental_mib": float(row["incremental_peak_bytes"]) / 2**20,
            }
        )
    return pd.DataFrame(rows)


def build_repeated_audit(root: Path, diagnosis: Path, audit: dict | None) -> None:
    sources = {
        "efficiency benchmark": root / "efficiency.py",
        "LiteFusion model": root / "kd_litefusion_mkan_teacher/model.py",
        "MKAN reproduction model": root / "mkan_refine/reproduction/model.py",
    }
    lines = [
        "# Repeated-operation audit",
        "",
        "This audit separates operations present in model source from operations actually repeated inside one timed forward.",
        "",
        "## Static source scan",
        "",
        "| Source | Operation | Source lines |",
        "|---|---|---|",
    ]
    for name, path in sources.items():
        result = scan_source(path)
        for operation, source_lines in result.items():
            lines.append(
                f"| {name} | `{operation}` | {', '.join(map(str, source_lines)) if source_lines else 'none'} |"
            )
    lines.extend(
        [
            "",
            "## Runtime call-count audit",
            "",
        ]
    )
    if audit is None:
        lines.append("GPU component profile pending; runtime call counts are not yet available.")
    else:
        lines.extend(
            [
                "The hooks count the lowest common CLIP entry points: vision `conv1`, vision transformer, text embedding, and text transformer.",
                "",
                "| Model | Batch | Path | vision conv1 | vision transformer | text embedding | text transformer |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for model_key, model in audit["models"].items():
            for batch, values in model["batches"].items():
                for path_key, label in (
                    ("native_clip_call_counts_per_forward", "current native"),
                    ("canonical_clip_call_counts_per_forward", "canonical shared"),
                ):
                    counts = values[path_key]
                    lines.append(
                        f"| {model_key} | {batch} | {label} | {counts['vision_conv1']} | "
                        f"{counts['vision_transformer']} | {counts['text_embedding']} | {counts['text_transformer']} |"
                    )
    lines.extend(
        [
            "",
            "## Static conclusions",
            "",
            "- The GPU-tensor benchmark does not include PIL decoding, tokenizer work, a DataLoader, disk I/O, `.cpu()`, `.numpy()`, or `.item()` inside the timed model forward.",
            "- LiteFusion calls `encode_image` and `encode_text`; MKAN uses manual token-level extraction. These are different CLIP code paths and must not be treated as a proven shared CLIP cost.",
            "- LiteFusion performs explicit FP32 conversion and two input L2 normalizations. Their isolated latency is measured by the component profiler.",
            "- The old benchmark measured models sequentially in one process. The loop-local `module` wrapper survived `del runtime`, retaining a previous runtime and explaining the unequal memory baselines. Fresh-process isolation is required.",
        ]
    )
    (diagnosis / "repeated_operation_audit.md").write_text("\n".join(lines), encoding="utf-8")


def build_path_report(root: Path, diagnosis: Path, raw: dict, components: pd.DataFrame | None) -> None:
    lines = [
        "# Benchmark path comparison",
        "",
        "## Old preliminary paths",
        "",
        "| Model | GPU-tensor end-to-end path | CLIP calls | Head input |",
        "|---|---|---|---|",
        "| Single/ensemble MKAN | GPU images/tokens → manual token-level visual and text transformer extraction → MKAN head(s) → logits | one visual + one text transformer | 577 vision tokens, 77 text tokens, two global features |",
        "| LiteFusion | GPU images/tokens → `encode_image` + `encode_text` → normalize → Fusion/Gate/Classifier → logits | one image + one text encode | two global features |",
        "",
        "Although both use the same checkpoint, these are not identical CLIP execution paths. MKAN additionally materializes and projects all token features.",
        "",
        "## Timing boundaries used by the diagnosis",
        "",
        "- **Current native GPU tensor:** pre-created GPU image tensor and token IDs → each model's current implementation → logits.",
        "- **Canonical shared token encoder:** the exact same manual token-level image/text encoder → method-specific head → logits.",
        "- **Deployment raw image/text:** in-memory raw PIL images and raw strings → identical CLIP preprocess/tokenize → host-to-device → current native model → logits. Disk I/O is intentionally excluded and reported separately from the GPU-tensor boundary.",
        "- Every model runs in a fresh process, preventing CUDA allocations or Python wrapper references from leaking across methods.",
        "",
    ]
    if components is None:
        lines.append("Component measurements are pending an idle GPU.")
    else:
        lines.extend(
            [
                "## Measured end-to-end paths",
                "",
                "| Model | Batch | Current native ms | Canonical shared ms | Deployment ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model_key in ("teacher_single", "teacher_ensemble", "student_shared"):
            for batch in (1, 8):
                subset = components[(components.model_key == model_key) & (components.batch_size == batch)]
                def get(path):
                    row = subset[(subset.path == path) & (subset.component == "end_to_end")]
                    return float(row.iloc[0].latency_ms_mean)
                lines.append(
                    f"| {model_key} | {batch} | {get('current_native_gpu_tensor'):.3f} | "
                    f"{get('canonical_shared_token_encoder'):.3f} | {get('deployment_raw_image_text'):.3f} |"
                )
    (diagnosis / "benchmark_path_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def build_anomaly_report(diagnosis: Path, raw: dict, components: pd.DataFrame | None) -> None:
    old = old_measurement_table(raw)
    def old_value(model, mode, batch=1):
        return float(old[(old.model_key == model) & (old["mode"] == mode) & (old.batch_size == batch)].iloc[0].latency)

    old_single = old_value("teacher_single", "end_to_end")
    old_student = old_value("student_shared", "end_to_end")
    old_single_head = old_value("teacher_single", "fusion_head_only")
    old_student_head = old_value("student_shared", "fusion_head_only")
    predicted = old_single + (old_student_head - old_single_head)
    unexplained = old_student - predicted
    lines = [
        "# End-to-end anomaly diagnosis",
        "",
        "## Status",
        "",
        "The 2026-07-17 result remains **preliminary_invalid_end_to_end_benchmark** and is excluded from final paper claims.",
        "",
        "## Quantified inconsistency in the old run",
        "",
        f"- Single MKAN end-to-end: {old_single:.3f} ms; head-only: {old_single_head:.3f} ms.",
        f"- LiteFusion end-to-end: {old_student:.3f} ms; head-only: {old_student_head:.3f} ms.",
        f"- If only the head differed, the expected LiteFusion time was about {predicted:.3f} ms, not {old_student:.3f} ms.",
        f"- Unexplained excess: {unexplained:.3f} ms per batch-1 forward.",
        "",
        "## Confirmed defects in the old benchmark",
        "",
        "1. MKAN and LiteFusion did not use the same CLIP implementation path: token-level manual extraction versus native global encoders.",
        "2. Models were run sequentially in one process without per-model process isolation.",
        "3. A loop-local wrapper reference survived model deletion, retaining earlier runtime modules. This is directly visible in unequal CUDA baseline allocations and invalidates absolute peak-memory comparison.",
        "4. The run did not record GPU clocks, temperature, competing processes, or per-stage times, so load/frequency changes cannot be ruled out.",
        "5. The large gap appeared outside the measured heads, but the old data cannot identify which CLIP stage caused it.",
        "",
    ]
    if components is None:
        lines.extend(
            [
                "## Pending measurement",
                "",
                "The new isolated component profile is waiting for an idle V100. No corrected latency is reported yet.",
            ]
        )
    else:
        canonical = components[
            (components.path == "canonical_shared_token_encoder")
            & (components.component.isin(["clip_image_encode", "clip_text_encode"]))
        ]
        totals = canonical.groupby(["model_key", "batch_size"]).latency_ms_mean.sum()
        lines.extend(["## Isolated rerun checks", ""])
        passed = True
        for batch in (1, 8):
            values = [float(totals.loc[(model, batch)]) for model in ("teacher_single", "teacher_ensemble", "student_shared")]
            spread = (max(values) - min(values)) / min(values) * 100.0
            passed = passed and spread <= 10.0
            lines.append(
                f"- Batch {batch} canonical shared CLIP range: {min(values):.3f}–{max(values):.3f} ms; spread {spread:.2f}% ({'PASS' if spread <= 10 else 'FAIL'})."
            )
        lines.extend(
            [
                "",
                f"Shared-CLIP <=10% acceptance rule: **{'PASS' if passed else 'FAIL'}**.",
                "Final efficiency publication remains blocked until the separate 50-warmup/200-iteration/5-round rebenchmark also passes.",
            ]
        )
    (diagnosis / "end_to_end_anomaly_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    diagnosis = args.diagnosis_dir.resolve()
    diagnosis.mkdir(parents=True, exist_ok=True)
    raw = json.loads(args.old_raw.read_text(encoding="utf-8"))
    component_path = diagnosis / "component_latency.csv"
    audit_path = diagnosis / "dtype_and_device_audit.json"
    components = pd.read_csv(component_path) if component_path.is_file() else None
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    build_repeated_audit(root, diagnosis, audit)
    build_path_report(root, diagnosis, raw, components)
    build_anomaly_report(diagnosis, raw, components)
    print(diagnosis / "end_to_end_anomaly_report.md")


if __name__ == "__main__":
    main()
