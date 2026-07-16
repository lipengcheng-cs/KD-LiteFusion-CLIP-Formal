# Formal matched three-seed summary

All values are mean ± sample standard deviation over three matched student seeds (3407, 42, 2024). Formal main results use `best_weighted_f1.pt`. Results from `best_macro_f1.pt` are supplementary only and are not mixed into the main table.

| Condition | Accuracy | Weighted-F1 | Macro-F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| wo_kd | 0.8811 ± 0.0129 | 0.8957 ± 0.0065 | 0.7590 ± 0.0103 | 0.9162 ± 0.0036 | 0.8811 ± 0.0129 |
| logits_kd | 0.9070 ± 0.0022 | 0.9075 ± 0.0020 | 0.8450 ± 0.0205 | 0.9087 ± 0.0015 | 0.9070 ± 0.0022 |

## Paired Logits-KD change

| Metric | Mean delta | SD | Positive seeds |
|---|---:|---:|---:|
| accuracy | +0.0260 | 0.0118 | 3/3 |
| weighted_f1 | +0.0118 | 0.0056 | 3/3 |
| macro_f1 | +0.0860 | 0.0107 | 3/3 |
| precision | -0.0075 | 0.0045 | 0/3 |
| recall | +0.0260 | 0.0118 | 3/3 |

`affected_individuals` has test support=7. Its per-class F1 is therefore highly unstable; a large single-run change is not evidence of a stable class-level gain.

Hyperparameter selection and later statistical diagnostics must use validation-only selection rules and must not retroactively select these test results.
