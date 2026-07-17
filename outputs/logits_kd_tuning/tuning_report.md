# Staged Logits KD tuning report

Selection used validation Weighted-F1 first and validation Macro-F1 second on development seed 3407. Test metrics were not loaded or used for hyperparameter selection.

## Stage A: temperature (weight=0.5)

| T | Validation Weighted-F1 | Validation Macro-F1 | Provenance |
|---:|---:|---:|---|
| 2.0 | 0.908092 | 0.911538 | trained |
| 4.0 | 0.906578 | 0.857351 | reused_complete |
| 6.0 | 0.908637 | 0.860780 | trained |

Selected temperature: **6.0**.

## Stage B: Logits KD weight

| Weight | Validation Weighted-F1 | Validation Macro-F1 | Provenance |
|---:|---:|---:|---|
| 0.25 | 0.906559 | 0.825753 | trained |
| 0.50 | 0.908637 | 0.860780 | reused_complete |
| 1.00 | 0.911630 | 0.872784 | trained |

Final selected configuration: **T=6.0, logits_kd_weight=1.00**.

This tuning result is validation-selected and does not replace the original formal three-seed result. If it differs from T=4.0, weight=0.5, it must be evaluated in a separate `formal_multiseed_tuned` directory.
