#!/usr/bin/env python3
"""Allowed LiteFusion-v2 checks: random head smoke, one real train batch, static MACs."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kd_litefusion_mkan_teacher.data import build_dataloaders
from kd_litefusion_mkan_teacher.litefusion_v2 import CANDIDATE_NAMES, LiteFusionV2Model, load_config
from kd_litefusion_mkan_teacher.litefusion_v2.profiling import parameter_breakdown, static_head_macs
from kd_litefusion_mkan_teacher.utils import move_to_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/litefusion_v2")
    parser.add_argument("--csv-path", default="data/clean/task2_clean_consistent.csv")
    parser.add_argument("--image-root", default="data/CrisisMMD_v2.0")
    parser.add_argument("--output-dir", default="outputs/litefusion_v2/audit")
    parser.add_argument("--real-candidate", default="v2_b_balanced", choices=CANDIDATE_NAMES)
    return parser.parse_args()


def shapes(outputs: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
    return {key: list(value.shape) for key, value in outputs.items()}


def check_contract(outputs: Dict[str, torch.Tensor], batch_size: int, feature_dim: int, num_classes: int) -> None:
    expected = {
        "logits": [batch_size, num_classes],
        "feature": [batch_size, feature_dim],
        "gate": [batch_size, feature_dim],
        "image_feature": [batch_size, feature_dim],
        "text_feature": [batch_size, feature_dim],
    }
    actual = shapes(outputs)
    if actual != expected:
        raise AssertionError(f"Output contract mismatch: actual={actual}, expected={expected}")
    if "vision_feature" in outputs:
        raise AssertionError("LiteFusion-v2 must not expose vision_feature")


def main() -> None:
    args = parse_args()
    set_seed(3407)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()

    random_smoke: List[Dict[str, Any]] = []
    static_rows: List[Dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        config = load_config(str(Path(args.config_dir) / f"{candidate}.yaml"))
        model = LiteFusionV2Model(config, load_clip=False).eval()
        image_feature = torch.randn(2, config.feature_dim)
        text_feature = torch.randn(2, config.feature_dim)
        with torch.inference_mode():
            outputs = model.forward_head(image_feature, text_feature, return_dict=True)
        check_contract(outputs, 2, config.feature_dim, config.num_classes)
        params = parameter_breakdown(model)
        macs = static_head_macs(model, batch_size=1)
        random_smoke.append(
            {
                "candidate": candidate,
                "status": "ok",
                "shapes": shapes(outputs),
                "finite": all(bool(torch.isfinite(value).all()) for value in outputs.values()),
            }
        )
        static_rows.append(
            {
                "candidate": candidate,
                "interaction_rank": config.interaction_rank,
                "residual_rank": config.residual_rank,
                "gate_type": config.gate_type,
                "gate_hidden": config.gate_hidden,
                "gate_groups": config.gate_groups,
                "classifier_hidden": config.classifier_hidden,
                "fusion_params": params["fusion"],
                "gate_params": params["gate"],
                "classifier_params": params["classifier"],
                "head_params": params["full_head"],
                "fusion_macs_batch1": macs["fusion"],
                "gate_macs_batch1": macs["gate"],
                "classifier_macs_batch1": macs["classifier"],
                "head_macs_batch1": macs["full_head"],
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_config = load_config(str(Path(args.config_dir) / f"{args.real_candidate}.yaml"))
    real_model = LiteFusionV2Model(real_config, device=device, load_clip=True).to(device).eval()
    loaders, _, _, teacher_cache = build_dataloaders(
        csv_path=args.csv_path,
        image_root=args.image_root,
        preprocess=real_model.preprocess,
        batch_size=1,
        num_workers=0,
        teacher_cache_path=None,
    )
    if teacher_cache is not None:
        raise AssertionError("Audit smoke must not attach a teacher cache")
    if "train" not in loaders or "val" not in loaders:
        raise ValueError("Audit requires explicit train and val splits")
    batch = next(iter(loaders["train"]))
    sample_ids = list(batch["sample_id"])
    batch = move_to_device(batch, device)
    with torch.inference_mode():
        real_outputs = real_model(batch["images"], batch["text_tokens"], return_dict=True)
    check_contract(real_outputs, 1, real_config.feature_dim, real_config.num_classes)
    if real_model.clip.training:
        raise AssertionError("Frozen CLIP left eval mode")
    real_smoke = {
        "candidate": args.real_candidate,
        "status": "ok",
        "split": "train",
        "sample_id": sample_ids,
        "device": str(device),
        "shapes": shapes(real_outputs),
        "finite": all(bool(torch.isfinite(value).all()) for value in real_outputs.values()),
        "clip_training": bool(real_model.clip.training),
    }

    metadata = {
        "git_commit_before_changes": commit,
        "seed": 3407,
        "kd": False,
        "test_evaluation": False,
        "device": str(device),
    }
    with (output_dir / "random_feature_smoke.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": random_smoke}, handle, indent=2, ensure_ascii=False)
    with (output_dir / "real_batch_smoke.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "result": real_smoke}, handle, indent=2, ensure_ascii=False)
    with (output_dir / "tensor_shape_trace.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "random_feature": random_smoke,
                "real_batch": real_smoke,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with (output_dir / "static_profiling.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": static_rows}, handle, indent=2, ensure_ascii=False)
    with (output_dir / "static_profiling.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(static_rows[0]))
        writer.writeheader()
        writer.writerows(static_rows)

    print(json.dumps({"random_feature_smoke": random_smoke, "real_batch_smoke": real_smoke}, indent=2))
    for row in static_rows:
        print(
            f"{row['candidate']}: head_params={row['head_params']}, "
            f"head_macs_batch1={row['head_macs_batch1']}"
        )


if __name__ == "__main__":
    main()
