# Formal matched multiseed completion report

Overall status: **PASS**

Expected fixed split: train=6090, val=995, test=950.

| Condition | Seed | Status | Failed checks |
|---|---:|---|---|
| wo_kd | 3407 | PASS | - |
| wo_kd | 42 | PASS | - |
| wo_kd | 2024 | PASS | - |
| logits_kd | 3407 | PASS | - |
| logits_kd | 42 | PASS | - |
| logits_kd | 2024 | PASS | - |

Formal-teacher Logits KD is accepted only when the formal cache check report is PASS. The w/o-KD runs must not load any teacher cache.
