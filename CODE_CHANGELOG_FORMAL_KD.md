# Formal KD Code Changelog

## 2026-07-14 — Stage 1 data-contract audit

- Created the required timestamped `backup_before_formal_kd_*` directory before edits.
- Added `scripts/audit_teacher_student_data_contract.py`.
- Added `scripts/10_audit_data_contract.sh`.
- The audit is read-only with respect to source datasets and writes only under
  `outputs/formal_kd_stage/` and `logs/formal_kd_stage/`.
- Existing baseline, Logits KD, and reproduction-teacher artifacts are not overwritten.
- The existing teacher is consistently identified as the MKAN-Refine supplied-source
  reproduction teacher, not an author-original checkpoint or strict B-spline reproduction.

## 2026-07-16 — New-server recovery and storage-safe retry

- Restored and checksum-verified the protected historical teacher, w/o KD, and Logits KD
  result directories without overwriting their contents.
- Initialized a compact Git repository that excludes datasets, weights, checkpoints,
  caches, logs, and server backup directories.
- Diagnosed the formal teacher run stopping at seed 42 epoch 13: DataLoader workers
  failed to create multiprocessing resources after the shared root filesystem ran out
  of space.
- Stopped only the confirmed hung tmux run; retained every completed seed 3407 artifact.
- Changed formal teacher training `num_workers` from 4 to 0 to remove multiprocessing
  `/tmp` dependence and avoid duplicating the multi-gigabyte cached tensors in workers.
- Added strict completed-seed detection to `train_formal_teacher.py`. A seed is skipped
  only when every required artifact exists and its checkpoint identity and training
  protocol match the formal supplied-source reproduction teacher.
- The incomplete seed 42 is intentionally retrained; completed seed 3407 is protected.
- Existing `backup_before_*` directories are not used for new changes and will only be
  removed after the Git repository has been pushed successfully to GitHub.
