# Matched multiseed: w/o KD vs Logits KD

All primary results use each run's `best_weighted_f1.pt`. The best-Macro-F1
checkpoints are preserved only as a separate sensitivity analysis.

| Condition | Accuracy | Weighted-F1 | Macro-F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| wo_kd | 0.8811 ± 0.0129 | 0.8957 ± 0.0065 | 0.7590 ± 0.0103 | 0.9162 ± 0.0036 | 0.8811 ± 0.0129 |
| logits_kd | 0.9070 ± 0.0022 | 0.9075 ± 0.0020 | 0.8450 ± 0.0205 | 0.9087 ± 0.0015 | 0.9070 ± 0.0022 |

The only intended matched-pair difference is whether formal-teacher Logits KD
is enabled. Data, split, seed, batch size, epochs, rank, optimizer, learning rate,
class weighting, label smoothing, and checkpoint rules are fixed.
