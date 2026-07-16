# Code changelog: next formal KD stage

## Formal matched multiseed completion and summary

- Added strict six-run completion validation covering artifacts, fixed split counts, seed/rank/frozen-CLIP configuration, formal-cache usage, w/o-KD isolation, finite outputs, checkpoint readability, and test prediction counts.
- Added a dry-run-first recovery entry point. Complete runs emit `SKIP_COMPLETE`; partial directories are archived before any explicitly authorized recovery run.
- Added publication-facing three-seed summaries with paired Logits-KD deltas and a strict distinction between the primary `best_weighted_f1.pt` checkpoint and supplementary `best_macro_f1.pt` sensitivity results.
- Recorded the `affected_individuals` support=7 stability warning.

The teacher is named **MKAN-Refine supplied-source reproduction teacher** (基于现有源码重训的 MKAN-Refine 复现教师). It is not an author-original checkpoint and is not described as a true B-spline KAN.
