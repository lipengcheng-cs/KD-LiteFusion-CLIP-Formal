# Code changelog: next formal KD stage

## Formal matched multiseed completion and summary

- Added strict six-run completion validation covering artifacts, fixed split counts, seed/rank/frozen-CLIP configuration, formal-cache usage, w/o-KD isolation, finite outputs, checkpoint readability, and test prediction counts.
- Added a dry-run-first recovery entry point. Complete runs emit `SKIP_COMPLETE`; partial directories are archived before any explicitly authorized recovery run.
- Added publication-facing three-seed summaries with paired Logits-KD deltas and a strict distinction between the primary `best_weighted_f1.pt` checkpoint and supplementary `best_macro_f1.pt` sensitivity results.
- Recorded the `affected_individuals` support=7 stability warning.

The teacher is named **MKAN-Refine supplied-source reproduction teacher** (基于现有源码重训的 MKAN-Refine 复现教师). It is not an author-original checkpoint and is not described as a true B-spline KAN.

## Staged Logits KD tuning

- Added a validation-only two-stage search: temperature `{2, 4, 6}` at weight 0.5, followed by weight `{0.25, 0.5, 1.0}` at the selected temperature.
- Reused the already-complete T=4, weight=0.5 formal seed-3407 run and the selected Stage-A trial instead of retraining identical configurations.
- Enforced seed 3407, rank 32, frozen CLIP, `num_workers=0`, fixed split, formal PASS teacher logits, independent trial directories, and finite/complete artifacts.
- Selected T=6.0 and logits weight=1.0 by validation Weighted-F1 with validation Macro-F1 as the tie-breaker. Test metrics were not used for tuning.
- Preserved the original formal three-seed results; tuned three-seed confirmation must use `outputs/formal_multiseed_tuned/`.

## Formal class-level statistical analysis

- Added sample-id-aligned paired prediction transitions, per-class metrics, per-seed and averaged normalized confusion matrices, all seven affected test cases, and rescue regression/improvement case tables.
- Added exact paired McNemar tests for every seed.
- Added a fixed-seed, class-stratified, paired 2,000-replicate bootstrap for Accuracy, Weighted-F1, Macro-F1, paired deltas, and per-class F1 with support recorded.
- Kept all statistical analysis strictly post-selection; no statistic is used to choose a model or hyperparameter.
- Recorded positive 95% paired intervals for Weighted-F1 and Macro-F1 and the support=7 limitation for `affected_individuals`.
