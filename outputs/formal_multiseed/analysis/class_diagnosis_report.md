# Formal KD class-level statistical diagnosis

All analyses use fixed formal test predictions after model selection. These statistics were not used to choose a model or hyperparameter.

## Paired significance

| Seed | w/o correct, KD wrong | w/o wrong, KD correct | Exact McNemar p |
|---:|---:|---:|---:|
| 3407 | 12 | 27 | 0.0237027 |
| 42 | 12 | 49 | 1.96988e-06 |
| 2024 | 7 | 29 | 0.000312551 |

## Three-seed mean paired bootstrap delta

| Metric | Mean | 95% CI |
|---|---:|---:|
| accuracy | +0.0262 | [+0.0179, +0.0351] |
| weighted_f1 | +0.0120 | [+0.0051, +0.0190] |
| macro_f1 | +0.0864 | [+0.0477, +0.1231] |

## Small-support and rescue checks

`affected_individuals` has exactly 7 test samples. Its F1 interval and apparent gains must be interpreted as highly support-sensitive.

Rescue regressions by seed: {42: 3, 2024: 4, 3407: 6}. Rescue improvements by seed: {42: 9, 2024: 3, 3407: 4}.

Per-seed class metrics, transition tables, confusion matrices, exact paired tests, and fixed-seed 2,000-replicate stratified bootstrap intervals are stored alongside this report.
