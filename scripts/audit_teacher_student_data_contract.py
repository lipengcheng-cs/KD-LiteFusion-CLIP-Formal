#!/usr/bin/env python3
"""Audit the MKAN teacher and LiteFusion student data contracts.

This script is intentionally read-only with respect to source datasets. It records
all overlaps, duplicates, label agreements, label merges, and split leakage needed
to decide whether an existing teacher is suitable for formal KD experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


TEACHER_FILES = {
    "train": "task02_train.tsv",
    "val": "task02_dev.tsv",
    "test": "task02_test.tsv",
}
EXPECTED_TEACHER_COUNTS = {"train": 5119, "val": 1097, "test": 1098}
EXPECTED_STUDENT_COUNTS = {"train": 6090, "val": 995, "test": 950}
TEACHER_NATIVE_ORDER = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]
STUDENT_FIXED_ORDER = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian",
]
LABEL_MERGE = {
    "injured_or_dead_people": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "vehicle_damage": "infrastructure_and_utility_damage",
}


def canonical_label(value: object) -> str:
    label = str(value).strip()
    return LABEL_MERGE.get(label, label)


def duplicate_report(df: pd.DataFrame, id_col: str) -> List[str]:
    ids = df[id_col].astype(str)
    return sorted(ids[ids.duplicated(keep=False)].unique().tolist())


def count_dict(series: pd.Series, splits: Iterable[str]) -> Dict[str, int]:
    counts = series.astype(str).str.lower().value_counts().to_dict()
    return {split: int(counts.get(split, 0)) for split in splits}


def load_teacher(data_dir: Path) -> Dict[str, pd.DataFrame]:
    splits: Dict[str, pd.DataFrame] = {}
    for split, filename in TEACHER_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, sep="\t")
        required = {"image_id", "label", "label_id"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame = df.copy()
        frame["sample_id"] = frame["image_id"].astype(str)
        frame["split"] = split
        frame["raw_label"] = frame["label"].astype(str)
        frame["canonical_label"] = frame["raw_label"].map(canonical_label)
        splits[split] = frame
    return splits


def load_student(path: Path) -> Dict[str, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"sample_id", "label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["split"].astype(str).str.lower()
    df["raw_label"] = df["label"].astype(str)
    df["canonical_label"] = df["raw_label"].map(canonical_label)
    unknown_splits = sorted(set(df["split"]) - {"train", "val", "test"})
    if unknown_splits:
        raise ValueError(f"Unknown student splits: {unknown_splits}")
    return {split: df[df["split"] == split].copy() for split in ("train", "val", "test")}


def build_markdown(report: Dict) -> str:
    matrix = report["overlap_matrix"]
    lines = [
        "# Teacher–Student Data Contract Audit",
        "",
        f"- Audit status: **{report['status']}**",
        f"- Existing teacher eligibility: **{report['existing_teacher_eligibility']}**",
        f"- Formal fixed-split teacher required: **{report['formal_teacher_required']}**",
        f"- Teacher train ∩ student val: **{report['critical_overlaps']['teacher_train_student_val']}**",
        f"- Teacher train ∩ student test: **{report['critical_overlaps']['teacher_train_student_test']}**",
        f"- Canonical label conflicts: **{report['label_conflict_count']}**",
        "",
        "## 3×3 overlap matrix",
        "",
        "| teacher \\ student | train | val | test |",
        "|---|---:|---:|---:|",
    ]
    for teacher_split in ("train", "val", "test"):
        row = matrix[teacher_split]
        lines.append(
            f"| {teacher_split} | {row['train']} | {row['val']} | {row['test']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "## Contract details",
            "",
            f"- Teacher counts: `{report['teacher_counts']}`",
            f"- Student counts: `{report['student_counts']}`",
            f"- Teacher native class order: `{report['teacher_native_class_order']}`",
            f"- Student fixed class order: `{report['student_fixed_class_order']}`",
            f"- Off-diagonal split overlaps: `{report['off_diagonal_overlap_count']}`",
            f"- Duplicate sample IDs: `{report['duplicate_summary']}`",
            f"- Legacy-label merge counts: `{report['legacy_label_merge_counts']}`",
            "",
            "All overlapping samples are written to `data_overlap_samples.csv`; canonical",
            "label conflicts are written to `label_conflicts.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-data-dir", type=Path, required=True)
    parser.add_argument("--student-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    teacher = load_teacher(args.teacher_data_dir)
    student = load_student(args.student_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    teacher_counts = {split: len(df) for split, df in teacher.items()}
    student_counts = {split: len(df) for split, df in student.items()}
    teacher_duplicates = {
        split: duplicate_report(df, "sample_id") for split, df in teacher.items()
    }
    student_duplicates = {
        split: duplicate_report(df, "sample_id") for split, df in student.items()
    }

    overlap_matrix: Dict[str, Dict[str, int]] = {}
    overlap_rows: List[Dict] = []
    for teacher_split, teacher_df in teacher.items():
        overlap_matrix[teacher_split] = {}
        teacher_lookup = teacher_df.set_index("sample_id", drop=False)
        for student_split, student_df in student.items():
            student_lookup = student_df.set_index("sample_id", drop=False)
            overlap_ids = sorted(set(teacher_lookup.index) & set(student_lookup.index))
            overlap_matrix[teacher_split][student_split] = len(overlap_ids)
            for sample_id in overlap_ids:
                teacher_row = teacher_lookup.loc[sample_id]
                student_row = student_lookup.loc[sample_id]
                if isinstance(teacher_row, pd.DataFrame) or isinstance(student_row, pd.DataFrame):
                    raise ValueError(f"Duplicate ID prevents unambiguous comparison: {sample_id}")
                teacher_raw = str(teacher_row["raw_label"])
                student_raw = str(student_row["raw_label"])
                teacher_canonical = str(teacher_row["canonical_label"])
                student_canonical = str(student_row["canonical_label"])
                overlap_rows.append(
                    {
                        "sample_id": sample_id,
                        "teacher_split": teacher_split,
                        "student_split": student_split,
                        "teacher_raw_label": teacher_raw,
                        "student_raw_label": student_raw,
                        "teacher_canonical_label": teacher_canonical,
                        "student_canonical_label": student_canonical,
                        "raw_label_match": teacher_raw == student_raw,
                        "canonical_label_match": teacher_canonical == student_canonical,
                        "cross_split": teacher_split != student_split,
                    }
                )

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_columns = [
        "sample_id", "teacher_split", "student_split", "teacher_raw_label",
        "student_raw_label", "teacher_canonical_label", "student_canonical_label",
        "raw_label_match", "canonical_label_match", "cross_split",
    ]
    if overlap_df.empty:
        overlap_df = pd.DataFrame(columns=overlap_columns)
    overlap_df = overlap_df[overlap_columns].sort_values(
        ["teacher_split", "student_split", "sample_id"]
    )
    conflicts_df = overlap_df[~overlap_df["canonical_label_match"].astype(bool)].copy()

    matrix_df = pd.DataFrame.from_dict(overlap_matrix, orient="index")
    matrix_df = matrix_df.loc[["train", "val", "test"], ["train", "val", "test"]]
    matrix_df.index.name = "teacher_split"
    matrix_df.to_csv(args.output_dir / "data_overlap_matrix.csv")
    overlap_df.to_csv(args.output_dir / "data_overlap_samples.csv", index=False)
    conflicts_df.to_csv(args.output_dir / "label_conflicts.csv", index=False)

    teacher_train_student_val = overlap_matrix["train"]["val"]
    teacher_train_student_test = overlap_matrix["train"]["test"]
    off_diagonal = sum(
        overlap_matrix[t][s]
        for t in ("train", "val", "test")
        for s in ("train", "val", "test")
        if t != s
    )
    duplicate_total = sum(len(v) for v in teacher_duplicates.values()) + sum(
        len(v) for v in student_duplicates.values()
    )
    count_mismatch = (
        teacher_counts != EXPECTED_TEACHER_COUNTS or student_counts != EXPECTED_STUDENT_COUNTS
    )
    leakage = teacher_train_student_val > 0 or teacher_train_student_test > 0
    formal_teacher_required = off_diagonal > 0 or count_mismatch
    blocking_issues = leakage or duplicate_total > 0 or len(conflicts_df) > 0 or count_mismatch
    status = "FAIL_CURRENT_TEACHER_PROTOCOL" if blocking_issues else "PASS"
    eligibility = "EXPLORATORY_ONLY" if leakage else "NO_TRAIN_TO_EVAL_LEAKAGE_DETECTED"
    if leakage:
        decision = (
            "The existing reproduction teacher and its current Logits KD result are exploratory only. "
            "Teacher training includes samples assigned to the student's validation and/or test split. "
            "Do not use those results as formal paper evidence. Train a new supplied-source reproduction "
            "teacher strictly on the fixed student train split before formal KD."
        )
    elif blocking_issues:
        decision = (
            "The audit found a blocking data-contract issue. Stop formal KD until the listed duplicate, "
            "label, or count problem is resolved."
        )
    elif formal_teacher_required:
        decision = (
            "No teacher-train to student-evaluation leakage was found, but split contracts differ. "
            "Train the formal fixed-split reproduction teacher before continuing."
        )
    else:
        decision = "The audited split and label contracts pass the formal KD gate."

    all_student = pd.concat(student.values(), ignore_index=True)
    legacy_counts = {
        legacy: int((all_student["raw_label"] == legacy).sum()) for legacy in LABEL_MERGE
    }
    observed_teacher_order = (
        pd.concat(teacher.values(), ignore_index=True)[["label_id", "raw_label"]]
        .drop_duplicates()
        .sort_values("label_id")["raw_label"]
        .astype(str)
        .tolist()
    )
    report = {
        "status": status,
        "existing_teacher_eligibility": eligibility,
        "formal_teacher_required": bool(formal_teacher_required),
        "decision": decision,
        "teacher_data_dir": str(args.teacher_data_dir.resolve()),
        "student_csv": str(args.student_csv.resolve()),
        "teacher_counts": teacher_counts,
        "student_counts": student_counts,
        "expected_teacher_counts": EXPECTED_TEACHER_COUNTS,
        "expected_student_counts": EXPECTED_STUDENT_COUNTS,
        "count_mismatch": bool(count_mismatch),
        "overlap_matrix": overlap_matrix,
        "critical_overlaps": {
            "teacher_train_student_val": teacher_train_student_val,
            "teacher_train_student_test": teacher_train_student_test,
        },
        "off_diagonal_overlap_count": int(off_diagonal),
        "total_overlap_records": int(len(overlap_df)),
        "label_conflict_count": int(len(conflicts_df)),
        "duplicate_summary": {
            "teacher": {k: len(v) for k, v in teacher_duplicates.items()},
            "student": {k: len(v) for k, v in student_duplicates.items()},
        },
        "duplicate_sample_ids": {
            "teacher": teacher_duplicates,
            "student": student_duplicates,
        },
        "label_merge": LABEL_MERGE,
        "legacy_label_merge_counts": legacy_counts,
        "teacher_native_class_order": TEACHER_NATIVE_ORDER,
        "teacher_observed_class_order": observed_teacher_order,
        "student_fixed_class_order": STUDENT_FIXED_ORDER,
        "class_order_matches": TEACHER_NATIVE_ORDER == STUDENT_FIXED_ORDER,
        "artifacts": {
            "overlap_matrix": "data_overlap_matrix.csv",
            "all_overlap_samples": "data_overlap_samples.csv",
            "label_conflicts": "label_conflicts.csv",
        },
    }
    (args.output_dir / "data_contract_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "data_contract_audit.md").write_text(
        build_markdown(report), encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "existing_teacher_eligibility": eligibility,
        "formal_teacher_required": bool(formal_teacher_required),
        "critical_overlaps": report["critical_overlaps"],
        "off_diagonal_overlap_count": int(off_diagonal),
        "label_conflict_count": int(len(conflicts_df)),
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
