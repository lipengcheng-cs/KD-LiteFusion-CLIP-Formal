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
