#!/usr/bin/env python3
"""Strictly validate the fixed-split formal teacher cache gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


EXPECTED_LABELS = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "rescue_volunteering_or_donation_effort": 2,
    "other_relevant_information": 3,
    "not_humanitarian": 4,
}
IDENTITY = "mkan_refine_supplied_source_reproduction"
PROTOCOL = "student_fixed_split_6090_995_950"


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--logits-cache", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checks = []

    def check(name, condition, detail):
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    df = pd.read_csv(args.csv)
    expected_ids = df.loc[df["split"].astype(str).str.lower() == "train", "sample_id"].astype(str).tolist()
    strict = load_torch(args.logits_cache)
    full = load_torch(args.full_cache)
    strict_ids = [str(value) for value in strict.get("sample_ids", [])]
    full_ids = [str(value) for value in full.get("sample_ids", [])]
    check("expected_train_count", len(expected_ids) == 6090, str(len(expected_ids)))
    check("strict_unique_ids", len(strict_ids) == 6090 and len(set(strict_ids)) == 6090, str(len(set(strict_ids))))
    check("strict_exact_id_order", strict_ids == expected_ids, "sample_id exact order")
    check("full_exact_id_order", full_ids == expected_ids, "sample_id exact order")
    check("strict_logits_shape", tuple(strict.get("logits", torch.empty(0)).shape) == (6090, 5), str(tuple(strict.get("logits", torch.empty(0)).shape)))
    check("full_logits_shape", tuple(full.get("logits", torch.empty(0)).shape) == (6090, 5), str(tuple(full.get("logits", torch.empty(0)).shape)))
    check("feature_shape", tuple(full.get("feature", torch.empty(0)).shape) == (6090, 768), str(tuple(full.get("feature", torch.empty(0)).shape)))
    check("gate_shape", tuple(full.get("gate", torch.empty(0)).shape) == (6090, 768), str(tuple(full.get("gate", torch.empty(0)).shape)))
    check("prototype_shape", tuple(full.get("prototypes", torch.empty(0)).shape) == (5, 768), str(tuple(full.get("prototypes", torch.empty(0)).shape)))
    tensors = [strict.get("logits"), full.get("logits"), full.get("feature"), full.get("gate"), full.get("prototypes")]
    check("all_finite", all(isinstance(value, torch.Tensor) and torch.isfinite(value).all().item() for value in tensors), "logits/feature/gate/prototypes")
    check("label_mapping", strict.get("label_to_id") == EXPECTED_LABELS and full.get("label_to_id") == EXPECTED_LABELS, str(strict.get("label_to_id")))
    check("teacher_identity", strict.get("teacher_identity") == IDENTITY and full.get("teacher_identity") == IDENTITY, str(strict.get("teacher_identity")))
    check("strict_b_spline_false", strict.get("strict_b_spline_reproduction") is False and full.get("strict_b_spline_reproduction") is False, "must be false")
    check("training_protocol", strict.get("training_protocol") == PROTOCOL and full.get("training_protocol") == PROTOCOL, str(strict.get("training_protocol")))
    check("audit_path_present", bool(strict.get("data_audit_report")) and Path(strict["data_audit_report"]).is_file(), str(strict.get("data_audit_report")))
    check("checkpoints_present", len(strict.get("teacher_checkpoints", [])) in (1, 3) and all(Path(path).is_file() for path in strict.get("teacher_checkpoints", [])), str(strict.get("teacher_checkpoints")))
    check("strategy_present", bool(strict.get("selected_teacher_strategy")), str(strict.get("selected_teacher_strategy")))
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    report = {"status": status, "checks": checks, "logits_cache": str(args.logits_cache.resolve()), "full_cache": str(args.full_cache.resolve())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
