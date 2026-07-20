#!/usr/bin/env python3
"""Diagnose formal teacher gate behavior without training a Gate-KD loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ModalityDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, preprocess):
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.preprocess = preprocess

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        import clip

        row = self.frame.iloc[index]
        path = Path(str(row.image_path))
        if not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as image:
            pixels = self.preprocess(image.convert("RGB"))
        tokens = clip.tokenize(str(row.text), truncate=True).squeeze(0)
        return index, pixels, tokens


def compute_modality_difference(frame: pd.DataFrame, image_root: Path) -> np.ndarray:
    """Stream frozen CLIP encodings; do not regenerate the deleted 7.6GB cache."""
    import clip

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for streamed image/text difference diagnosis")
    device = torch.device("cuda")
    model, preprocess = clip.load("/home/lpc/.cache/clip/ViT-L-14-336px.pt", device=device, jit=False)
    model.eval()
    dataset = ModalityDataset(frame, image_root, preprocess)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    differences = np.empty(len(dataset), dtype=np.float32)
    with torch.no_grad():
        for indices, images, tokens in loader:
            vision = F.normalize(model.encode_image(images.to(device)).float(), dim=-1)
            text = F.normalize(model.encode_text(tokens.to(device)).float(), dim=-1)
            diff = torch.linalg.vector_norm(vision - text, dim=-1).cpu().numpy()
            differences[indices.numpy()] = diff
    del model
    torch.cuda.empty_cache()
    if not np.isfinite(differences).all():
        raise RuntimeError("Streamed modality differences contain NaN/Inf")
    return differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    cache_root = root / "outputs" / "server_mkan_kd_formal" / "teacher_cache"
    report = json.loads((cache_root / "check_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("Formal teacher full cache check report must be PASS")
    cache = torch.load(cache_root / "mkan_train_full.pt", map_location="cpu", weights_only=False)
    gate = cache["gate"].float().numpy()
    feature = cache["feature"].float().numpy()
    logits = cache["logits"].float().numpy()
    sample_ids = [str(x) for x in cache["sample_ids"]]
    if gate.shape != (6090, 768) or not np.isfinite(gate).all():
        raise RuntimeError(f"Invalid gate tensor: {gate.shape}")
    source = pd.read_csv(cache["source_csv"])
    source = source[source.split.astype(str).str.lower() == "train"].copy()
    source.sample_id = source.sample_id.astype(str)
    source = source.set_index("sample_id").loc[sample_ids].reset_index()
    label_to_id = {str(k): int(v) for k, v in cache["label_to_id"].items()}
    labels = source.label.map(label_to_id).to_numpy(dtype=int)
    prediction = logits.argmax(axis=1)
    correct = prediction == labels
    output = root / "outputs" / "gate_analysis"
    output.mkdir(parents=True, exist_ok=True)

    dimension = pd.DataFrame({
        "dimension": np.arange(gate.shape[1]),
        "mean": gate.mean(axis=0),
        "variance": gate.var(axis=0),
        "std": gate.std(axis=0),
        "min": gate.min(axis=0),
        "max": gate.max(axis=0),
        "near_zero_fraction_le_0p05": (gate <= 0.05).mean(axis=0),
        "near_one_fraction_ge_0p95": (gate >= 0.95).mean(axis=0),
    })
    dimension.to_csv(output / "gate_dimension_statistics.csv", index=False)
    sample_gate_mean = gate.mean(axis=1)
    sample_gate_std = gate.std(axis=1)
    class_rows = []
    for label, class_id in sorted(label_to_id.items(), key=lambda item: item[1]):
        mask = labels == class_id
        class_rows.append({
            "class_id": class_id,
            "class_name": label,
            "support": int(mask.sum()),
            "gate_element_mean": float(gate[mask].mean()),
            "gate_element_std": float(gate[mask].std()),
            "sample_gate_mean_mean": float(sample_gate_mean[mask].mean()),
            "sample_gate_mean_std": float(sample_gate_mean[mask].std()),
            "sample_gate_std_mean": float(sample_gate_std[mask].mean()),
            "near_zero_fraction_le_0p05": float((gate[mask] <= 0.05).mean()),
            "near_one_fraction_ge_0p95": float((gate[mask] >= 0.95).mean()),
        })
    class_stats = pd.DataFrame(class_rows)
    class_stats.to_csv(output / "gate_class_statistics.csv", index=False)
    saturation = pd.DataFrame([
        {"threshold": "<=0.01", "fraction": float((gate <= 0.01).mean())},
        {"threshold": "<=0.05", "fraction": float((gate <= 0.05).mean())},
        {"threshold": ">=0.95", "fraction": float((gate >= 0.95).mean())},
        {"threshold": ">=0.99", "fraction": float((gate >= 0.99).mean())},
        {"threshold": "outside_[0.05,0.95]", "fraction": float(((gate <= 0.05) | (gate >= 0.95)).mean())},
    ])
    saturation.to_csv(output / "gate_saturation.csv", index=False)
    correctness_rows = []
    for value, name in ((True, "teacher_correct"), (False, "teacher_incorrect")):
        mask = correct == value
        correctness_rows.append({"group": name, "support": int(mask.sum()), "sample_gate_mean_mean": float(sample_gate_mean[mask].mean()), "sample_gate_mean_std": float(sample_gate_mean[mask].std()), "sample_gate_std_mean": float(sample_gate_std[mask].mean()), "gate_element_mean": float(gate[mask].mean()), "gate_element_std": float(gate[mask].std())})
    pd.DataFrame(correctness_rows).to_csv(output / "gate_correct_vs_incorrect.csv", index=False)

    # The full formal cache deliberately contains fused feature and gate only. Stream
    # frozen CLIP encodings without saving them, so the deleted 7.6GB cache stays deleted.
    modality_difference = compute_modality_difference(source, root / "data" / "CrisisMMD_v2.0")
    modality_corr = float(np.corrcoef(sample_gate_mean, modality_difference)[0, 1])
    feature_norm = np.linalg.norm(feature, axis=1)
    fused_corr = float(np.corrcoef(sample_gate_mean, feature_norm)[0, 1])
    correlation = pd.DataFrame([
        {"analysis": "image_text_feature_difference_vs_gate", "status": "AVAILABLE_STREAMED", "correlation": modality_corr, "n": 6090, "reason": "Frozen CLIP image/text features were streamed with num_workers=0 and were not saved; the deleted 7.6GB cache was not regenerated."},
        {"analysis": "fused_teacher_feature_norm_vs_sample_gate_mean", "status": "AVAILABLE_SUPPLEMENTARY", "correlation": fused_corr, "n": 6090, "reason": "Supplementary diagnostic only; not a modality-difference substitute."},
    ])
    correlation.to_csv(output / "gate_feature_difference_correlation.csv", index=False)

    overall_std = float(gate.std())
    median_dim_std = float(dimension["std"].median())
    saturation_fraction = float(((gate <= 0.05) | (gate >= 0.95)).mean())
    class_mean_range = float(class_stats.sample_gate_mean_mean.max() - class_stats.sample_gate_mean_mean.min())
    near_constant = bool(overall_std < 0.05 or median_dim_std < 0.02)
    severely_saturated = bool(saturation_fraction >= 0.5)
    sufficient_for_gate_kd_discussion = bool(not near_constant and not severely_saturated and class_mean_range >= 0.01)
    summary = {
        "status": "PASS_DIAGNOSTIC",
        "teacher_name": "MKAN-Refine supplied-source reproduction teacher",
        "teacher_identity": cache.get("teacher_identity"),
        "strict_b_spline_reproduction": cache.get("strict_b_spline_reproduction"),
        "gate_generation": {
            "method": "validation-selected weighted ensemble of per-seed gates",
            "teacher_checkpoints": cache.get("teacher_checkpoints"),
            "ensemble_weights": cache.get("ensemble_weights"),
            "selected_teacher_strategy": cache.get("selected_teacher_strategy"),
        },
        "shape": list(gate.shape),
        "overall_mean": float(gate.mean()),
        "overall_std": overall_std,
        "overall_variance": float(gate.var()),
        "median_dimension_std": median_dim_std,
        "saturation_outside_0p05_0p95": saturation_fraction,
        "class_sample_gate_mean_range": class_mean_range,
        "near_constant_thresholds": {"overall_std_min": 0.05, "median_dimension_std_min": 0.02},
        "near_constant": near_constant,
        "severely_saturated_threshold": 0.5,
        "severely_saturated": severely_saturated,
        "sufficient_for_future_gate_kd_discussion": sufficient_for_gate_kd_discussion,
        "image_text_difference_correlation_available": True,
        "image_text_difference_gate_correlation": modality_corr,
        "affected_individuals_support": int((labels == label_to_id["affected_individuals"]).sum()),
        "rescue_support": int((labels == label_to_id["rescue_volunteering_or_donation_effort"]).sum()),
    }
    (output / "gate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plt.figure(figsize=(7.2, 4.6))
    plt.hist(gate.ravel(), bins=80, color="#3b82f6", alpha=0.85)
    plt.xlabel("Gate value")
    plt.ylabel("Element count")
    plt.title("Formal teacher gate distribution")
    plt.tight_layout()
    plt.savefig(output / "gate_distribution_overall.png", dpi=180)
    plt.close()
    plt.figure(figsize=(9.5, 5.0))
    values = [sample_gate_mean[labels == class_id] for _, class_id in sorted(label_to_id.items(), key=lambda item: item[1])]
    names = [label.replace("_", "\n") for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    plt.boxplot(values, labels=names, showfliers=False)
    plt.ylabel("Per-sample mean gate")
    plt.title("Formal teacher mean gate by class")
    plt.xticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "gate_distribution_by_class.png", dpi=180)
    plt.close()

    lines = [
        "# Formal teacher gate diagnostic",
        "",
        "This is diagnosis only. No Gate-KD loss was implemented or trained.",
        "",
        f"The gate cache is a validation-selected weighted ensemble of three supplied-source reproduction teacher seeds with weights `{cache.get('ensemble_weights')}`.",
        "",
        f"Overall gate mean={summary['overall_mean']:.6f}, std={overall_std:.6f}, median per-dimension std={median_dim_std:.6f}, and saturation outside [0.05, 0.95]={saturation_fraction:.2%}.",
        "",
        f"The diagnostic near-constant flag is **{near_constant}** and severe-saturation flag is **{severely_saturated}**. Class mean range={class_mean_range:.6f}. Under the documented exploratory thresholds, future Gate-KD discussion is **{'allowed' if sufficient_for_gate_kd_discussion else 'not supported'}**; this is not evidence that Gate KD will improve a student.",
        "",
        f"The image/text feature-difference correlation was computed by streaming frozen CLIP features with `num_workers=0` and without saving them (r={modality_corr:.6f}). The deleted 7.6GB modality cache was not regenerated.",
        "",
        "Class statistics include affected and rescue support. Correct-vs-incorrect teacher gate summaries and publication-ready CSV/PNG artifacts are stored in this directory.",
        "",
    ]
    (output / "gate_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
