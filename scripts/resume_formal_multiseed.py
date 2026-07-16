#!/usr/bin/env python3
"""Resume only missing/incomplete matched runs, preserving partial artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--execute", action="store_true", help="Without this flag, only report planned actions.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    checker = root / "scripts" / "check_formal_multiseed_completion.py"
    subprocess.run([args.python, str(checker), "--project-root", str(root)], check=False)
    report = json.loads((root / "outputs" / "formal_multiseed" / "completion_report.json").read_text(encoding="utf-8"))
    failed = [run for run in report["runs"] if run["status"] != "PASS"]
    if not failed:
        print("SKIP_COMPLETE: all six matched runs already PASS")
        return
    for run in failed:
        print(f"NEEDS_RECOVERY: {run['condition']} seed={run['seed']}")
    if not args.execute:
        print("DRY_RUN: pass --execute after reviewing the completion report")
        return
    active = subprocess.run(["pgrep", "-af", "train.py|12_run_matched_multiseed"], text=True, capture_output=True)
    if active.returncode == 0 and active.stdout.strip():
        raise RuntimeError("A matching training process is already active; refusing duplicate launch")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / "outputs" / "formal_multiseed"
    for run in failed:
        path = Path(run["run_dir"])
        if path.exists():
            archived = output_root / "incomplete_archives" / f"{run['condition']}_seed_{run['seed']}_{stamp}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archived))
            print(f"ARCHIVED_INCOMPLETE: {path} -> {archived}")
    subprocess.run(
        [args.python, str(root / "scripts" / "run_matched_multiseed_experiments.py"), "--project-root", str(root), "--python", args.python],
        cwd=root,
        check=True,
    )
    subprocess.run([args.python, str(checker), "--project-root", str(root)], check=True)


if __name__ == "__main__":
    main()
