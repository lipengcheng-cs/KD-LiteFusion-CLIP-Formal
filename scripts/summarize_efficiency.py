#!/usr/bin/env python3
"""Create formal CSVs, plots, and the fair efficiency report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


TEACHER = "MKAN-Refine supplied-source reproduction teacher"

EXPECTED_MODEL_KEYS = ("teacher_single", "teacher_ensemble", "student_shared")
EXPECTED_MODES = ("end_to_end", "fusion_head_only", "fusion_module_only")
EXPECTED_BATCHES = (1, 8)


def load_raw_with_duplicate_key_audit(path: Path) -> tuple[dict, list[str]]:
    duplicate_keys: list[str] = []

    def audit_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicate_keys.append(str(key))
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=audit_pairs), duplicate_keys


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def validate_raw_schema(raw: dict, duplicate_json_keys: list[str]) -> dict:
    missing_fields: list[dict] = []
    duplicate_records: list[dict] = []
    invalid_values: list[dict] = []
    warnings: list[str] = []

    def require(mapping: Any, fields: tuple[str, ...], path: str) -> None:
        if not isinstance(mapping, dict):
            missing_fields.append({"path": path, "field": "<mapping>", "detail": "expected object"})
            return
        for field in fields:
            if field not in mapping:
                missing_fields.append({"path": path, "field": field})

    require(
        raw,
        ("protocol", "environment", "models", "measurements", "flops", "student_checkpoints"),
        "raw",
    )
    protocol = raw.get("protocol", {})
    require(protocol, ("warmup", "iterations", "rounds", "batch_sizes", "clip_model"), "raw.protocol")
    environment = raw.get("environment", {})
    require(environment, ("timestamp", "pytorch", "pytorch_cuda", "gpu", "precision"), "raw.environment")

    models = raw.get("models", {})
    require(models, EXPECTED_MODEL_KEYS, "raw.models")
    model_numeric_fields = (
        "total_parameters",
        "trainable_parameters",
        "frozen_parameters",
        "clip_parameters",
        "fusion_head_parameters",
        "fusion_core_parameters",
        "fusion_plus_gate_parameters",
        "gate_parameters",
        "classifier_parameters",
        "checkpoint_size_bytes",
        "clip_checkpoint_size_bytes",
        "deployment_size_bytes",
        "heads_executed",
    )
    for model_key in EXPECTED_MODEL_KEYS:
        values = models.get(model_key)
        require(values, model_numeric_fields + ("checkpoint_paths",), f"raw.models.{model_key}")
        if not isinstance(values, dict):
            continue
        for field in model_numeric_fields:
            if field not in values:
                continue
            value = values[field]
            if not _is_finite_number(value) or float(value) < 0:
                invalid_values.append(
                    {"path": f"raw.models.{model_key}.{field}", "value": repr(value), "rule": "finite >= 0"}
                )
        if "checkpoint_paths" in values and not values["checkpoint_paths"]:
            invalid_values.append(
                {"path": f"raw.models.{model_key}.checkpoint_paths", "value": repr(values["checkpoint_paths"]), "rule": "non-empty"}
            )

    measurements = raw.get("measurements", [])
    if not isinstance(measurements, list) or not measurements:
        invalid_values.append({"path": "raw.measurements", "value": repr(measurements), "rule": "non-empty list"})
        measurements = []
    measurement_required = (
        "model_key",
        "mode",
        "batch_size",
        "latency_ms_rounds",
        "latency_ms_mean",
        "latency_ms_std",
        "throughput_samples_per_s_rounds",
        "throughput_samples_per_s_mean",
        "throughput_samples_per_s_std",
        "baseline_bytes",
        "peak_bytes",
        "incremental_peak_bytes",
    )
    seen_measurements: set[tuple] = set()
    for index, row in enumerate(measurements):
        path = f"raw.measurements[{index}]"
        require(row, measurement_required, path)
        if not isinstance(row, dict):
            continue
        key = (row.get("model_key"), row.get("mode"), row.get("batch_size"))
        if key in seen_measurements:
            duplicate_records.append({"collection": "measurements", "key": list(key)})
        seen_measurements.add(key)
        for field in (
            "latency_ms_mean",
            "throughput_samples_per_s_mean",
            "baseline_bytes",
            "peak_bytes",
            "incremental_peak_bytes",
        ):
            if field not in row:
                continue
            value = row[field]
            lower_bound = 0.0 if field.endswith("bytes") else np.nextafter(0.0, 1.0)
            if not _is_finite_number(value) or float(value) < lower_bound:
                invalid_values.append(
                    {"path": f"{path}.{field}", "value": repr(value), "rule": f"finite >= {lower_bound}"}
                )
        for field in ("latency_ms_std", "throughput_samples_per_s_std"):
            if field in row and (not _is_finite_number(row[field]) or float(row[field]) < 0):
                invalid_values.append(
                    {"path": f"{path}.{field}", "value": repr(row[field]), "rule": "finite >= 0"}
                )
        for field in ("latency_ms_rounds", "throughput_samples_per_s_rounds"):
            values = row.get(field)
            if not isinstance(values, list) or not values:
                invalid_values.append({"path": f"{path}.{field}", "value": repr(values), "rule": "non-empty list"})
            elif any(not _is_finite_number(value) or float(value) <= 0 for value in values):
                invalid_values.append({"path": f"{path}.{field}", "value": repr(values), "rule": "all finite > 0"})

    expected_measurements = {
        (model_key, mode, batch)
        for model_key in EXPECTED_MODEL_KEYS
        for mode in EXPECTED_MODES
        for batch in EXPECTED_BATCHES
    }
    for key in sorted(expected_measurements - seen_measurements):
        missing_fields.append({"path": "raw.measurements", "field": list(key), "detail": "missing record"})

    flops = raw.get("flops", [])
    if not isinstance(flops, list) or not flops:
        invalid_values.append({"path": "raw.flops", "value": repr(flops), "rule": "non-empty list"})
        flops = []
    flops_required = (
        "model_key",
        "mode",
        "flops_per_sample",
        "macs_per_sample_assuming_2_flops_per_mac",
        "method",
    )
    seen_flops: set[tuple] = set()
    for index, row in enumerate(flops):
        path = f"raw.flops[{index}]"
        require(row, flops_required, path)
        if not isinstance(row, dict):
            continue
        key = (row.get("model_key"), row.get("mode"))
        if key in seen_flops:
            duplicate_records.append({"collection": "flops", "key": list(key)})
        seen_flops.add(key)
        for field in ("flops_per_sample", "macs_per_sample_assuming_2_flops_per_mac"):
            if field in row and (not _is_finite_number(row[field]) or float(row[field]) <= 0):
                invalid_values.append(
                    {"path": f"{path}.{field}", "value": repr(row[field]), "rule": "finite > 0"}
                )
    expected_flops = {(model_key, mode) for model_key in EXPECTED_MODEL_KEYS for mode in EXPECTED_MODES}
    for key in sorted(expected_flops - seen_flops):
        missing_fields.append({"path": "raw.flops", "field": list(key), "detail": "missing record"})

    # These are semantic warnings: the raw values are retained for diagnosis, not publication.
    try:
        lookup = {(row["model_key"], row["mode"], int(row["batch_size"])): row for row in measurements}
        single = float(lookup[("teacher_single", "end_to_end", 1)]["latency_ms_mean"])
        student = float(lookup[("student_shared", "end_to_end", 1)]["latency_ms_mean"])
        single_head = float(lookup[("teacher_single", "fusion_head_only", 1)]["latency_ms_mean"])
        student_head = float(lookup[("student_shared", "fusion_head_only", 1)]["latency_ms_mean"])
        if abs(student - single) > 1.1 * abs(student_head - single_head):
            warnings.append(
                "End-to-end latency gap is inconsistent with the head-only gap; mark this run preliminary_invalid_end_to_end_benchmark."
            )
        baselines = [
            float(lookup[(model, "end_to_end", 1)]["baseline_bytes"])
            for model in EXPECTED_MODEL_KEYS
        ]
        if max(baselines) > min(baselines) * 1.2:
            warnings.append(
                "GPU baseline memory differs by more than 20% across models; absolute peak memory is not a fair cross-model comparison."
            )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        warnings.append("Semantic anomaly checks could not run because required measurement records are incomplete.")

    status = "PASS" if not (missing_fields or duplicate_json_keys or duplicate_records or invalid_values) else "FAIL"
    return {
        "status": status,
        "benchmark_status": "preliminary_invalid_end_to_end_benchmark",
        "counts": {
            "models": len(models) if isinstance(models, dict) else 0,
            "measurements": len(measurements),
            "flops_records": len(flops),
        },
        "missing_fields": missing_fields,
        "duplicate_json_keys": sorted(set(duplicate_json_keys)),
        "duplicate_records": duplicate_records,
        "invalid_values": invalid_values,
        "warnings": warnings,
    }


def write_schema_reports(report: dict, output_dir: Path) -> None:
    (output_dir / "schema_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Efficiency raw benchmark schema check",
        "",
        f"- Schema status: **{report['status']}**",
        f"- Benchmark publication status: **{report['benchmark_status']}**",
        f"- Models: {report['counts']['models']}",
        f"- Measurement records: {report['counts']['measurements']}",
        f"- FLOPs records: {report['counts']['flops_records']}",
        "",
    ]
    for title, key in (
        ("Missing fields or records", "missing_fields"),
        ("Duplicate JSON fields", "duplicate_json_keys"),
        ("Duplicate records", "duplicate_records"),
        ("Invalid numeric or empty values", "invalid_values"),
        ("Semantic warnings", "warnings"),
    ):
        lines.extend([f"## {title}", ""])
        values = report[key]
        if values:
            lines.extend(f"- `{json.dumps(value, ensure_ascii=False)}`" for value in values)
        else:
            lines.append("- None")
        lines.append("")
    (output_dir / "schema_check.md").write_text("\n".join(lines), encoding="utf-8")


def preserve_or_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Keep an existing raw-derived CSV byte-for-byte when its data is equivalent."""
    if path.is_file():
        try:
            existing = pd.read_csv(path)
            pd.testing.assert_frame_equal(
                existing.reset_index(drop=True),
                frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
            return
        except (AssertionError, ValueError, pd.errors.ParserError):
            recovered = path.with_name(f"{path.stem}_recovered{path.suffix}")
            frame.to_csv(recovered, index=False)
            return
    frame.to_csv(path, index=False)


def require_index_labels(frame: pd.DataFrame, labels: list[str], context: str) -> None:
    missing = [label for label in labels if label not in frame.index]
    if missing:
        raise ValueError(
            f"{context} is missing methods {missing}; available methods are {list(frame.index)}. "
            "See outputs/efficiency/schema_check.json for the raw schema audit."
        )


def font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def short_method(name: str) -> str:
    if "best single" in name:
        return "MKAN single"
    if "3-model" in name:
        return "MKAN ensemble"
    if "w/o KD" in name:
        return "LiteFusion w/o KD"
    if "Logits + Feature" in name:
        return "LiteFusion Logits+Feature"
    if "Feature KD" in name:
        return "LiteFusion Feature KD"
    return "LiteFusion Logits KD"


def draw_scatter(frame: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, path: Path) -> None:
    width, height = 2400, 1500
    left, right, top, bottom = 260, 100, 100, 260
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis_font, label_font = font(42), font(34)
    x_values = frame[x_col].astype(float).to_numpy()
    y_values = frame[y_col].astype(float).to_numpy()
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_pad = (x_max - x_min) * 0.08 or max(abs(x_min) * 0.05, 1.0)
    y_pad = (y_max - y_min) * 0.12 or max(abs(y_min) * 0.05, 0.01)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    x0, x1, y0, y1 = left, width - right, height - bottom, top
    draw.line((x0, y0, x1, y0), fill="black", width=4)
    draw.line((x0, y0, x0, y1), fill="black", width=4)
    for index in range(6):
        fraction = index / 5
        px = x0 + fraction * (x1 - x0)
        py = y0 - fraction * (y0 - y1)
        draw.line((px, y0, px, y1), fill="#dddddd", width=2)
        draw.line((x0, py, x1, py), fill="#dddddd", width=2)
        xv = x_min + fraction * (x_max - x_min)
        yv = y_min + fraction * (y_max - y_min)
        draw.text((px - 70, y0 + 20), f"{xv:.3g}", fill="black", font=label_font)
        draw.text((25, py - 22), f"{yv:.3f}", fill="black", font=label_font)
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
    for index, (_, row) in enumerate(frame.iterrows()):
        px = x0 + (float(row[x_col]) - x_min) / (x_max - x_min) * (x1 - x0)
        py = y0 - (float(row[y_col]) - y_min) / (y_max - y_min) * (y0 - y1)
        color = colors[index % len(colors)]
        draw.ellipse((px - 18, py - 18, px + 18, py + 18), fill=color, outline="black", width=3)
        draw.text((px + 24, py - 45 + (index % 2) * 46), short_method(row["Method"]), fill=color, font=label_font)
    draw.text(((x0 + x1) / 2 - 220, height - 90), xlabel, fill="black", font=axis_font)
    draw.text((25, 25), ylabel, fill="black", font=axis_font)
    image.save(path, dpi=(300, 300))


def draw_normalized_bars(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    width, height = 2600, 1600
    left, right, top, bottom = 220, 80, 100, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis_font, label_font = font(42), font(31)
    x0, x1, y0, y1 = left, width - right, height - bottom, top
    maximum = max(1.05, float(frame[columns].to_numpy().max()) * 1.12)
    draw.line((x0, y0, x1, y0), fill="black", width=4)
    draw.line((x0, y0, x0, y1), fill="black", width=4)
    for index in range(6):
        value = maximum * index / 5
        py = y0 - value / maximum * (y0 - y1)
        draw.line((x0, py, x1, py), fill="#dddddd", width=2)
        draw.text((30, py - 20), f"{value:.2f}", fill="black", font=label_font)
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    group_width = (x1 - x0) / len(frame)
    bar_width = group_width / (len(columns) + 1)
    for row_index, (_, row) in enumerate(frame.iterrows()):
        group_start = x0 + row_index * group_width + bar_width * 0.5
        for col_index, column in enumerate(columns):
            value = float(row[column])
            px0 = group_start + col_index * bar_width
            px1 = px0 + bar_width * 0.8
            py = y0 - value / maximum * (y0 - y1)
            draw.rectangle((px0, py, px1, y0), fill=colors[col_index], outline="black", width=2)
        draw.text((group_start, y0 + 25), short_method(row["Method"]), fill="black", font=label_font)
    legend_x = x0 + 30
    for index, column in enumerate(columns):
        draw.rectangle((legend_x + index * 430, 25, legend_x + 35 + index * 430, 60), fill=colors[index])
        draw.text((legend_x + 48 + index * 430, 18), column, fill="black", font=label_font)
    draw.text((20, 75), "Cost normalized to best single MKAN teacher", fill="black", font=axis_font)
    image.save(path, dpi=(300, 300))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def human(value: float, unit: str = "") -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f}G{unit}"
    if value >= 1e6:
        return f"{value / 1e6:.3f}M{unit}"
    if value >= 1e3:
        return f"{value / 1e3:.3f}K{unit}"
    return f"{value:.3f}{unit}"


def percent_change(reference: float, student: float, lower_is_better: bool = True) -> float:
    if lower_is_better:
        return (reference - student) / reference * 100.0
    return (student - reference) / reference * 100.0


def metric_text(value: float, std: float | None = None) -> str:
    if std is None or not math.isfinite(std):
        return f"{value:.4f}"
    return f"{value:.4f} ± {std:.4f}"


def read_performance(root: Path, raw: dict) -> dict:
    teacher_single = json.loads(
        (root / "outputs/server_mkan_kd_formal/seed_3407/test_metrics.json").read_text(encoding="utf-8")
    )
    teacher_ensemble = json.loads(
        (root / "outputs/server_mkan_kd_formal/reports/ensemble_test_metrics.json").read_text(encoding="utf-8")
    )
    formal = pd.read_csv(root / "outputs/formal_multiseed/summary/overall_mean_std.csv").set_index("condition")
    result = {
        "teacher_single": {
            "accuracy": teacher_single["accuracy"],
            "weighted_f1": teacher_single["weighted_f1"],
            "macro_f1": teacher_single["macro_f1"],
        },
        "teacher_ensemble": {
            "accuracy": teacher_ensemble["accuracy"],
            "weighted_f1": teacher_ensemble["weighted_f1"],
            "macro_f1": teacher_ensemble["macro_f1"],
        },
    }
    for key, condition in (("student_wo", "wo_kd"), ("student_logits", "logits_kd")):
        row = formal.loc[condition]
        result[key] = {
            metric: float(row[f"{metric}_mean"])
            for metric in ("accuracy", "weighted_f1", "macro_f1")
        }
        result[key].update(
            {f"{metric}_std": float(row[f"{metric}_std"]) for metric in ("accuracy", "weighted_f1", "macro_f1")}
        )
    final = raw.get("final_student")
    if final:
        frame = pd.read_csv(final["performance_csv"])
        result["student_final"] = {
            metric: float(frame[metric].mean()) for metric in ("accuracy", "weighted_f1", "macro_f1")
        }
        result["student_final"].update(
            {f"{metric}_std": float(frame[metric].std(ddof=1)) for metric in ("accuracy", "weighted_f1", "macro_f1")}
        )
    return result


def main() -> None:
    args = parse_args()
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_benchmark.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Required raw benchmark is missing: {raw_path}")
    raw, duplicate_json_keys = load_raw_with_duplicate_key_audit(raw_path)
    schema_report = validate_raw_schema(raw, duplicate_json_keys)
    write_schema_reports(schema_report, out)
    if schema_report["status"] != "PASS":
        raise ValueError(
            "Efficiency raw benchmark failed schema validation; "
            f"missing={len(schema_report['missing_fields'])}, "
            f"duplicate_json_keys={len(schema_report['duplicate_json_keys'])}, "
            f"duplicate_records={len(schema_report['duplicate_records'])}, "
            f"invalid_values={len(schema_report['invalid_values'])}. "
            f"See {out / 'schema_check.json'}."
        )
    perf = read_performance(root, raw)

    final_label = None
    if raw.get("final_student"):
        selected = raw["final_student"]["selection"]
        final_label = (
            "LiteFusion-CLIP + Logits + Feature KD"
            if float(selected.get("logits_kd_weight", 0.0)) > 0
            else "LiteFusion-CLIP + Feature KD"
        )
    labels = {
        "teacher_single": f"{TEACHER} (best single seed 3407)",
        "teacher_ensemble": f"{TEACHER} (formal 3-model ensemble)",
        "student_wo": "LiteFusion-CLIP w/o KD",
        "student_logits": "LiteFusion-CLIP + Logits KD",
    }
    if final_label:
        labels["student_final"] = final_label
    model_source = {
        "teacher_single": "teacher_single",
        "teacher_ensemble": "teacher_ensemble",
        "student_wo": "student_shared",
        "student_logits": "student_shared",
        "student_final": "student_shared",
    }

    method_keys = list(labels)
    parameter_rows = []
    for key in method_keys:
        values = raw["models"][model_source[key]]
        parameter_rows.append({"method": labels[key], **{k: v for k, v in values.items() if not isinstance(v, list)}})
    parameter_frame = pd.DataFrame(parameter_rows)
    preserve_or_write_csv(parameter_frame, out / "model_parameter_breakdown.csv")

    flops_lookup = {(row["model_key"], row["mode"]): row for row in raw["flops"]}
    flops_rows = []
    for key in method_keys:
        source = model_source[key]
        e2e = flops_lookup[(source, "end_to_end")]
        head = flops_lookup[(source, "fusion_head_only")]
        fusion = flops_lookup[(source, "fusion_module_only")]
        flops_rows.append(
            {
                "method": labels[key],
                "total_flops_per_sample": e2e["flops_per_sample"],
                "total_macs_per_sample": e2e["macs_per_sample_assuming_2_flops_per_mac"],
                "fusion_head_flops_per_sample": head["flops_per_sample"],
                "fusion_head_macs_per_sample": head["macs_per_sample_assuming_2_flops_per_mac"],
                "fusion_module_flops_per_sample": fusion["flops_per_sample"],
                "fusion_module_macs_per_sample": fusion["macs_per_sample_assuming_2_flops_per_mac"],
                "profiling_method": e2e["method"],
            }
        )
    flops_frame = pd.DataFrame(flops_rows)
    preserve_or_write_csv(flops_frame, out / "flops_macs_report.csv")

    measurement = {(row["model_key"], row["mode"], int(row["batch_size"])): row for row in raw["measurements"]}
    latency_frames = {}
    for batch in (1, 8):
        rows = []
        for key in method_keys:
            source = model_source[key]
            e2e = measurement[(source, "end_to_end", batch)]
            head = measurement[(source, "fusion_head_only", batch)]
            fusion = measurement[(source, "fusion_module_only", batch)]
            rows.append(
                {
                    "method": labels[key],
                    "batch_size": batch,
                    "end_to_end_latency_ms_mean": e2e["latency_ms_mean"],
                    "end_to_end_latency_ms_std": e2e["latency_ms_std"],
                    "fusion_head_latency_ms_mean": head["latency_ms_mean"],
                    "fusion_head_latency_ms_std": head["latency_ms_std"],
                    "fusion_module_latency_ms_mean": fusion["latency_ms_mean"],
                    "fusion_module_latency_ms_std": fusion["latency_ms_std"],
                    "measurement_rounds": raw["protocol"]["rounds"],
                    "iterations_per_round": raw["protocol"]["iterations"],
                }
            )
        frame = pd.DataFrame(rows)
        preserve_or_write_csv(frame, out / f"latency_batch{batch}.csv")
        latency_frames[batch] = frame.set_index("method")

    throughput_rows, memory_rows = [], []
    for key in method_keys:
        source = model_source[key]
        for batch in (1, 8):
            for mode in ("end_to_end", "fusion_head_only"):
                row = measurement[(source, mode, batch)]
                throughput_rows.append(
                    {
                        "method": labels[key],
                        "mode": mode,
                        "batch_size": batch,
                        "throughput_samples_per_s_mean": row["throughput_samples_per_s_mean"],
                        "throughput_samples_per_s_std": row["throughput_samples_per_s_std"],
                    }
                )
                memory_rows.append(
                    {
                        "method": labels[key],
                        "mode": mode,
                        "batch_size": batch,
                        "peak_gpu_memory_bytes": row["peak_bytes"],
                        "peak_gpu_memory_mib": row["peak_bytes"] / 2**20,
                        "incremental_peak_memory_bytes": row["incremental_peak_bytes"],
                        "incremental_peak_memory_mib": row["incremental_peak_bytes"] / 2**20,
                    }
                )
    throughput_frame = pd.DataFrame(throughput_rows)
    preserve_or_write_csv(throughput_frame, out / "throughput.csv")
    memory_frame = pd.DataFrame(memory_rows)
    preserve_or_write_csv(memory_frame, out / "gpu_memory.csv")

    checkpoint_rows = []
    for key in method_keys:
        source_values = raw["models"][model_source[key]]
        if key.startswith("student_"):
            if key == "student_wo":
                path = Path(raw["student_checkpoints"]["wo_kd"])
            elif key == "student_logits":
                path = Path(raw["student_checkpoints"]["logits_kd"])
            else:
                path = Path(raw["final_student"]["checkpoint"])
            checkpoint_size = path.stat().st_size
            paths = str(path)
        else:
            checkpoint_size = source_values["checkpoint_size_bytes"]
            paths = ";".join(source_values["checkpoint_paths"])
        checkpoint_rows.append(
            {
                "method": labels[key],
                "checkpoint_paths": paths,
                "checkpoint_size_bytes": checkpoint_size,
                "checkpoint_size_mib": checkpoint_size / 2**20,
                "shared_clip_checkpoint_bytes": source_values["clip_checkpoint_size_bytes"],
                "deployment_size_bytes": source_values["clip_checkpoint_size_bytes"] + checkpoint_size,
            }
        )
    preserve_or_write_csv(pd.DataFrame(checkpoint_rows), out / "checkpoint_sizes.csv")

    param_index = parameter_frame.set_index("method")
    flops_index = flops_frame.set_index("method")
    # Bracket access is mandatory: ``DataFrame.mode`` resolves to the mode() method,
    # which previously produced an empty selection and a misleading KeyError.
    tp8 = throughput_frame[
        (throughput_frame["mode"] == "end_to_end") & (throughput_frame["batch_size"] == 8)
    ].set_index("method")
    mem8 = memory_frame[
        (memory_frame["mode"] == "end_to_end") & (memory_frame["batch_size"] == 8)
    ].set_index("method")
    expected_labels = [labels[key] for key in method_keys]
    require_index_labels(tp8, expected_labels, "Batch-8 end-to-end throughput table")
    require_index_labels(mem8, expected_labels, "Batch-8 end-to-end memory table")
    trade_rows = []
    for key in method_keys:
        name = labels[key]
        p = perf[key]
        trade_rows.append(
            {
                "Method": name,
                "Accuracy": p["accuracy"],
                "Accuracy_std": p.get("accuracy_std", np.nan),
                "Weighted-F1": p["weighted_f1"],
                "Weighted-F1_std": p.get("weighted_f1_std", np.nan),
                "Macro-F1": p["macro_f1"],
                "Macro-F1_std": p.get("macro_f1_std", np.nan),
                "Params": param_index.loc[name, "total_parameters"],
                "Trainable_Params": param_index.loc[name, "trainable_parameters"],
                "Fusion_Head_Params": param_index.loc[name, "fusion_head_parameters"],
                "FLOPs": flops_index.loc[name, "total_flops_per_sample"],
                "Fusion_Head_FLOPs": flops_index.loc[name, "fusion_head_flops_per_sample"],
                "Latency": latency_frames[1].loc[name, "end_to_end_latency_ms_mean"],
                "Latency_std": latency_frames[1].loc[name, "end_to_end_latency_ms_std"],
                "Fusion_Only_Latency": latency_frames[1].loc[name, "fusion_head_latency_ms_mean"],
                "Throughput": tp8.loc[name, "throughput_samples_per_s_mean"],
                "Throughput_std": tp8.loc[name, "throughput_samples_per_s_std"],
                "GPU_Memory": mem8.loc[name, "peak_gpu_memory_mib"],
            }
        )
    trade = pd.DataFrame(trade_rows)
    preserve_or_write_csv(trade, out / "performance_efficiency_tradeoff.csv")

    trade_index = trade.set_index("Method")
    relative_rows = []
    for student_key in [key for key in method_keys if key.startswith("student_")]:
        student = trade_index.loc[labels[student_key]]
        for reference_key in ("teacher_single", "teacher_ensemble"):
            reference = trade_index.loc[labels[reference_key]]
            relative_rows.append(
                {
                    "student": labels[student_key],
                    "reference": labels[reference_key],
                    "parameter_reduction_pct": percent_change(reference.Params, student.Params),
                    "trainable_head_parameter_reduction_pct": percent_change(reference.Trainable_Params, student.Trainable_Params),
                    "flops_reduction_pct": percent_change(reference.FLOPs, student.FLOPs),
                    "batch1_latency_reduction_pct": percent_change(reference.Latency, student.Latency),
                    "batch8_throughput_improvement_pct": percent_change(reference.Throughput, student.Throughput, False),
                    "batch8_peak_gpu_memory_reduction_pct": percent_change(reference.GPU_Memory, student.GPU_Memory),
                }
            )
    relative = pd.DataFrame(relative_rows)
    preserve_or_write_csv(relative, out / "relative_reduction_rates.csv")

    # Plot source CSVs and 300-DPI figures.
    plot_data = trade[["Method", "Weighted-F1", "Macro-F1", "Params", "FLOPs", "Latency"]].copy()
    preserve_or_write_csv(plot_data, out / "paper_efficiency_plot_data.csv")
    plots = [
        ("Params", "Weighted-F1", "weighted_f1_vs_parameters.png", "Total parameters", "Weighted-F1"),
        ("Latency", "Macro-F1", "macro_f1_vs_latency.png", "Batch-1 end-to-end latency (ms)", "Macro-F1"),
        ("FLOPs", "Weighted-F1", "performance_vs_flops.png", "End-to-end FLOPs per sample", "Weighted-F1"),
    ]
    for x, y, filename, xlabel, ylabel in plots:
        draw_scatter(plot_data, x, y, xlabel, ylabel, out / filename)

    compare_columns = ["Params", "FLOPs", "Latency", "GPU_Memory"]
    cost = trade.set_index("Method")[compare_columns].copy()
    normalized = cost.divide(cost.loc[labels["teacher_single"]], axis=1)
    normalized.insert(0, "Method", normalized.index)
    preserve_or_write_csv(normalized, out / "teacher_student_efficiency_comparison_data.csv")
    draw_normalized_bars(normalized, compare_columns, out / "teacher_student_efficiency_comparison.png")

    student_name = labels["student_logits"]
    single_rel = relative[(relative.student == student_name) & (relative.reference == labels["teacher_single"])].iloc[0]
    ensemble_rel = relative[(relative.student == student_name) & (relative.reference == labels["teacher_ensemble"])].iloc[0]
    wo_perf, kd_perf = perf["student_wo"], perf["student_logits"]
    student_params = param_index.loc[student_name]
    single_params = param_index.loc[labels["teacher_single"]]
    student_flops = flops_index.loc[student_name]
    single_flops = flops_index.loc[labels["teacher_single"]]

    def reduction_sentence(row):
        return (
            f"参数 {row.parameter_reduction_pct:+.2f}%、FLOPs {row.flops_reduction_pct:+.2f}%、"
            f"batch-1 延迟 {row.batch1_latency_reduction_pct:+.2f}%、batch-8 峰值显存 "
            f"{row.batch8_peak_gpu_memory_reduction_pct:+.2f}%，吞吐量变化 {row.batch8_throughput_improvement_pct:+.2f}%"
        )

    lines = [
        "# MKAN-Refine 复现教师与 LiteFusion 学生公平效率报告",
        "",
        "> **状态：preliminary_invalid_end_to_end_benchmark。** 本报告由已有 raw benchmark 恢复生成；原始端到端路径存在尚未解释的异常，只用于问题排查，不得作为论文最终效率结论。",
        "",
        f"测量时间：{raw['environment']['timestamp']}；GPU：{raw['environment']['gpu']}；PyTorch：{raw['environment']['pytorch']}；CUDA：{raw['environment']['pytorch_cuda']}；精度：FP32。",
        "",
        f"教师只能称为 **{TEACHER}（基于现有源码重训的 MKAN-Refine 复现教师）**。它不是作者原始 checkpoint，也不是严格 B-spline KAN。",
        "",
        "## 公平测量协议",
        "",
        f"所有方法使用同一块 V100、同一 OpenAI CLIP ViT-L/14@336px、本地权重、336×336 图像、77 token、`model.eval()` 与 `torch.inference_mode()`。每项先 warm-up {raw['protocol']['warmup']} 次，再计时 {raw['protocol']['iterations']} 次，共重复 {raw['protocol']['rounds']} 轮；每次调用前后均执行 `torch.cuda.synchronize()`。",
        "",
        "End-to-end 从图像张量与 token ids 开始，包含 CLIP、Fusion/Gate/Classifier，排除磁盘读取、图像解码和 tokenizer。Fusion/Head-only 从预计算 CLIP 特征开始，包含完整融合、门控和分类头。正式三模型集成共享一次 CLIP 编码，但三个 MKAN head 均实际执行，checkpoint 大小也按三者之和计算。",
        "",
        "FLOPs 来自 `torch.profiler(with_flops=True)`；未被 profiler 支持的逐元素算子可能未计入，因此报告同时保留方法说明，MACs 按 2 FLOPs≈1 MAC 换算。",
        "",
        "## 性能—效率主表",
        "",
        "| 方法 | Accuracy | Weighted-F1 | Macro-F1 | Total Params | Head Params | FLOPs | B1 latency (ms) | B8 throughput | B8 peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in method_keys:
        row = trade_index.loc[labels[key]]
        values = perf[key]
        lines.append(
            f"| {labels[key]} | {metric_text(values['accuracy'], values.get('accuracy_std'))} | "
            f"{metric_text(values['weighted_f1'], values.get('weighted_f1_std'))} | "
            f"{metric_text(values['macro_f1'], values.get('macro_f1_std'))} | "
            f"{human(row.Params)} | {human(row.Trainable_Params)} | {human(row.FLOPs)} | "
            f"{row.Latency:.3f} ± {row.Latency_std:.3f} | {row.Throughput:.2f} | {row.GPU_Memory:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 明确结论",
            "",
            f"- LiteFusion + Logits KD 相对最佳单 seed MKAN 复现教师：{reduction_sentence(single_rel)}。正值表示成本减少或吞吐提升，负值表示学生反而更高。",
            f"- LiteFusion + Logits KD 相对正式三模型集成：{reduction_sentence(ensemble_rel)}。集成成本按三个实际 head 计算，而非单 checkpoint。",
            f"- Logits KD 的 Weighted-F1 相对 w/o KD 提高 {kd_perf['weighted_f1'] - wo_perf['weighted_f1']:+.4f}，Macro-F1 提高 {kd_perf['macro_f1'] - wo_perf['macro_f1']:+.4f}。二者共享完全相同的学生推理结构和一次严格测量，推理不加载教师，因此 KD 没有增加推理成本。",
            f"- 低秩融合核心参数：LiteFusion {int(student_params.fusion_core_parameters):,}，单 MKAN cross-attention 核心 {int(single_params.fusion_core_parameters):,}；融合+Gate 参数分别为 {int(student_params.fusion_plus_gate_parameters):,} 与 {int(single_params.fusion_plus_gate_parameters):,}。完整 Fusion/Head-only FLOPs 分别为 {human(student_flops.fusion_head_flops_per_sample)} 与 {human(single_flops.fusion_head_flops_per_sample)}。因此是否‘真正轻量’必须按核心、融合+Gate、完整 head 三个口径分别判断，不能仅凭 low-rank 名称下结论。",
        ]
    )
    if final_label:
        lines.append(f"- 通过筛选的后续最终学生 `{final_label}` 已纳入；其结构仍与其他 LiteFusion 学生相同，效率复用同一次严格测量，性能来自验证选择后的三 seed 测试汇总。")
    else:
        lines.append("- Feature KD 尚未形成通过筛选且完成三 seed 的最终学生，因此本次未虚构第五种方法；后续产物满足条件时脚本会自动纳入。")
    lines.extend(
        [
            "",
            "## 输出说明",
            "",
            "所有原始轮次、标准差、参数拆分、checkpoint、FLOPs/MACs、batch-1/batch-8 延迟、吞吐和显存数据均保存在同目录 CSV/JSON 中。四张图均为 300 DPI，并有对应 CSV 数据。",
            "",
        ]
    )
    recovered_text = "\n".join(lines)
    (out / "efficiency_report.md").write_text(recovered_text, encoding="utf-8")
    (out / "efficiency_report_raw_recovered.md").write_text(recovered_text, encoding="utf-8")
    print(out / "efficiency_report_raw_recovered.md")


if __name__ == "__main__":
    main()
