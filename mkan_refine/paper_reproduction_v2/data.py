#!/usr/bin/env python3
"""Audit and load the fixed 7,314-sample MKAN paper protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


LABEL_TO_ID = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "not_humanitarian": 2,
    "other_relevant_information": 3,
    "rescue_volunteering_or_donation_effort": 4,
}
EXPECTED_COUNTS = {"train": 5119, "val": 1097, "test": 1098}
FILES = {"train": "task02_train.tsv", "val": "task02_dev.tsv", "test": "task02_test.tsv"}
REQUIRED_COLUMNS = {
    "tweet_id",
    "image_id",
    "label_text",
    "label_image",
    "tweet_text",
    "image_path",
    "label",
    "label_id",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-image-decode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir, image_root, output = (
        args.data_dir.resolve(),
        args.image_root.resolve(),
        args.output_dir.resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    file_records = []
    errors = []
    warnings = []

    for split, filename in FILES.items():
        path = data_dir / filename
        if not path.is_file():
            errors.append(f"missing TSV: {path}")
            continue
        frame = pd.read_csv(path, sep="\t", dtype={"tweet_id": str, "image_id": str})
        missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            errors.append(f"{split}: missing columns {missing_columns}")
            continue
        if len(frame) != EXPECTED_COUNTS[split]:
            errors.append(f"{split}: expected {EXPECTED_COUNTS[split]} rows, found {len(frame)}")
        frame = frame.copy()
        frame.insert(0, "split", split)
        frame.insert(1, "split_row", range(len(frame)))
        frame["sample_id"] = frame["image_id"].astype(str)
        frames.append(frame)
        file_records.append(
            {
                "split": split,
                "path": str(path),
                "rows": len(frame),
                "sha256": sha256(path),
            }
        )

    if not frames:
        raise RuntimeError(f"No valid TSV files; errors={errors}")
    all_data = pd.concat(frames, ignore_index=True)
    if len(all_data) != sum(EXPECTED_COUNTS.values()):
        errors.append(f"total: expected 7314 rows, found {len(all_data)}")

    duplicate_ids = all_data[all_data.duplicated("sample_id", keep=False)].sort_values("sample_id")
    cross_split_duplicate_ids = (
        all_data.groupby("sample_id")["split"].nunique().loc[lambda values: values > 1].index.tolist()
    )
    if not duplicate_ids.empty:
        errors.append(f"duplicate sample_id rows: {len(duplicate_ids)}")
    if cross_split_duplicate_ids:
        errors.append(f"cross-split duplicate sample_ids: {len(cross_split_duplicate_ids)}")

    label_ids = pd.to_numeric(all_data["label_id"], errors="coerce")
    invalid_label_id = label_ids.isna() | ~label_ids.isin(LABEL_TO_ID.values())
    expected_ids = all_data["label"].map(LABEL_TO_ID)
    mapping_mismatch = invalid_label_id | expected_ids.isna() | (label_ids != expected_ids)
    if mapping_mismatch.any():
        errors.append(f"label/label_id mapping mismatches: {int(mapping_mismatch.sum())}")

    text_image_mismatch = all_data["label_text"].astype(str) != all_data["label_image"].astype(str)
    final_text_mismatch = all_data["label"].astype(str) != all_data["label_text"].astype(str)
    final_image_mismatch = all_data["label"].astype(str) != all_data["label_image"].astype(str)
    if text_image_mismatch.any():
        errors.append(f"text/image label mismatches: {int(text_image_mismatch.sum())}")
    if final_text_mismatch.any() or final_image_mismatch.any():
        errors.append(
            f"final label disagreements: text={int(final_text_mismatch.sum())}, image={int(final_image_mismatch.sum())}"
        )

    image_exists = []
    image_nonempty = []
    image_decodes = []
    image_errors = []
    for relative in all_data["image_path"].astype(str):
        path = image_root / relative
        exists = path.is_file()
        nonempty = exists and path.stat().st_size > 0
        decodes = False
        error = ""
        if nonempty and not args.skip_image_decode:
            try:
                with Image.open(path) as image:
                    image.verify()
                decodes = True
            except Exception as exc:  # report exact corrupt files instead of substituting zero tensors
                error = f"{type(exc).__name__}: {exc}"
        elif nonempty:
            decodes = True
        image_exists.append(exists)
        image_nonempty.append(nonempty)
        image_decodes.append(decodes)
        image_errors.append(error)
    all_data["image_exists"] = image_exists
    all_data["image_nonempty"] = image_nonempty
    all_data["image_decodes"] = image_decodes
    all_data["image_error"] = image_errors
    missing_images = int((~all_data["image_exists"]).sum())
    empty_images = int((~all_data["image_nonempty"] & all_data["image_exists"]).sum())
    corrupt_images = int((~all_data["image_decodes"] & all_data["image_nonempty"]).sum())
    if missing_images or empty_images or corrupt_images:
        errors.append(
            f"image failures: missing={missing_images}, empty={empty_images}, decode_failed={corrupt_images}"
        )

    distribution = (
        all_data.groupby(["split", "label", "label_id"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    distribution["fraction"] = distribution["count"] / distribution.groupby("split")["count"].transform("sum")
    distribution.to_csv(output / "class_distribution.csv", index=False)

    manifest = all_data[
        [
            "split",
            "split_row",
            "sample_id",
            "tweet_id",
            "image_path",
            "label",
            "label_id",
            "image_exists",
            "image_nonempty",
            "image_decodes",
            "image_error",
        ]
    ].copy()
    manifest["text_sha256"] = all_data["tweet_text"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    manifest.to_csv(output / "split_manifest.csv", index=False)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "protocol": "MKAN paper protocol v2; fixed supplied TSV; no resplitting",
        "expected_counts": EXPECTED_COUNTS,
        "actual_counts": all_data.groupby("split").size().to_dict(),
        "total_rows": len(all_data),
        "files": file_records,
        "image_root": str(image_root),
        "image_checks": {
            "decode_enabled": not args.skip_image_decode,
            "missing": missing_images,
            "empty": empty_images,
            "decode_failed": corrupt_images,
        },
        "sample_id": {
            "field": "image_id",
            "unique": int(all_data["sample_id"].nunique()),
            "duplicate_rows": len(duplicate_ids),
            "cross_split_duplicate_ids": len(cross_split_duplicate_ids),
        },
        "label_order": LABEL_TO_ID,
        "label_checks": {
            "mapping_mismatches": int(mapping_mismatch.sum()),
            "text_image_mismatches": int(text_image_mismatch.sum()),
            "final_text_mismatches": int(final_text_mismatch.sum()),
            "final_image_mismatches": int(final_image_mismatch.sum()),
        },
        "student_8035_mixed": False,
        "errors": errors,
        "warnings": warnings,
    }
    (output / "data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# MKAN paper protocol v2 data audit",
        "",
        f"- Status: **{report['status']}**",
        "- Protocol: fixed supplied 5-class TSV files; no resplitting and no mixing with the 8,035-sample student dataset.",
        f"- Counts: train={report['actual_counts'].get('train', 0)}, val={report['actual_counts'].get('val', 0)}, test={report['actual_counts'].get('test', 0)}, total={report['total_rows']}.",
        f"- Unique sample IDs (`image_id`): {report['sample_id']['unique']}.",
        f"- Images: missing={missing_images}, empty={empty_images}, decode_failed={corrupt_images}.",
        f"- Label disagreements: text↔image={report['label_checks']['text_image_mismatches']}, final↔text={report['label_checks']['final_text_mismatches']}, final↔image={report['label_checks']['final_image_mismatches']}.",
        "",
        "## Fixed label order",
        "",
    ]
    lines.extend(f"- `{index}`: `{label}`" for label, index in LABEL_TO_ID.items())
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {error}" for error in errors] or ["- None"])
    lines.extend(["", "## File hashes", ""])
    lines.extend(f"- `{item['split']}`: `{item['sha256']}` — `{item['path']}`" for item in file_records)
    (output / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
