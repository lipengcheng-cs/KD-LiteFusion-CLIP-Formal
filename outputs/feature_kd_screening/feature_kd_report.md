# Feature KD screening report

Student alignment: final 768-dimensional fused feature after the Reliability-Aware Gate and before the classifier. Teacher alignment: 768-dimensional feature from the formal PASS full cache. Both are converted to FP32, L2-normalized, aligned by `sample_id`, and optimized with mean(1-cosine similarity); teacher features are detached.

Validation/test never read teacher cache. Selection used validation Weighted-F1 then validation Macro-F1 on seed 3407; test results were not used for selection.

| Condition | Feature weight | Validation Weighted-F1 | Validation Macro-F1 | Stop-rule failure |
|---|---:|---:|---:|---|
| feature_only | 0.05 | 0.899584 | 0.788962 | True |
| feature_only | 0.10 | 0.901495 | 0.796619 | True |
| feature_only | 0.20 | 0.900586 | 0.789546 | True |
| logits_feature | 0.05 | 0.911534 | 0.871785 | False |
| logits_feature | 0.10 | 0.911489 | 0.872592 | False |
| logits_feature | 0.20 | 0.913146 | 0.865281 | False |

Selected validation configuration: `{'condition': 'logits_feature', 'feature_kd_weight': 0.2, 'temperature': 6.0, 'logits_kd_weight': 1.0, 'selection_primary': 'validation_weighted_f1', 'selection_secondary': 'validation_macro_f1', 'test_used': False}`.

The selected configuration completed all three seeds. Test means±SD are reported only after validation selection:

| Metric | Mean ± SD |
|---|---:|
| accuracy | 0.9025 ± 0.0040 |
| weighted_f1 | 0.9035 ± 0.0035 |
| macro_f1 | 0.8517 ± 0.0257 |
| precision | 0.9062 ± 0.0022 |
| recall | 0.9025 ± 0.0040 |

The presence of a full teacher cache alone is not evidence that Feature KD is effective.
