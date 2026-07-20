# Efficiency and teacher revision changelog

## 2026-07-20 — Revision initiated

- Paused all new distillation extensions. No project-owned training or benchmark process was running when the revision started.
- Confirmed the GPU was occupied by another user, so no formal GPU measurement or training was launched.
- Created code-only backup `backup_before_efficiency_teacher_revision_20260720_175349/` (188 KiB). Existing checkpoints, caches, predictions, and experiment outputs were not altered.
- Located the efficiency summary failure: attribute access `throughput_frame.mode` and `memory_frame.mode` resolved to the Pandas `DataFrame.mode()` method instead of the `mode` column, producing an empty table followed by a model-name `KeyError`.
- Replaced ambiguous attribute access with explicit bracket access.
- Added raw JSON duplicate-key auditing; required-field and expected-record checks; duplicate measurement/FLOPs record checks; and validation for empty arrays, NaN, Inf, negative latency, negative memory, and non-positive throughput/FLOPs.
- Added explicit method-index checks so incomplete derived tables fail with actionable messages instead of a bare `KeyError`.
- Added `schema_check.json` and `schema_check.md` output.
- Added preservation logic that leaves equivalent existing raw-derived CSV files untouched.
- Added recovered report output `efficiency_report_raw_recovered.md` and marked the old run `preliminary_invalid_end_to_end_benchmark`.

Next required stage: run component-level path diagnosis. Formal rebenchmark remains blocked until the V100 has no obvious competing load.

## 2026-07-20 — MKAN paper protocol v2 supplement

- Uploaded the server-missing fixed paper TSV splits and verified SHA-256 equality with the supplied local files: 5,119 train, 1,097 validation, 1,098 test.
- Completed strict data audit: 7,314 unique sample IDs, zero cross-split duplicates, zero label disagreements, and 7,314/7,314 images present, non-empty, and decodable.
- Added paper/code alignment, metric protocol, consistency, assumptions, and baseline config documents under `outputs/mkan_paper_protocol_v2/`.
- Added independent `mkan_refine/paper_reproduction_v2/` route. Old supplied-source teacher code and results remain unchanged.
- Implemented real edge-wise B-spline KAN with SiLU base branch, recursive B-spline basis, learnable coefficients/scalers, grid buffer, FP32 finite checks, regularization, curve-to-coefficient fitting, and adaptive curve-preserving grid update.
- Implemented symmetric text→vision and vision→text KAN refinement, 768-dimensional KAN reliability gate, and KAN classifier.
- Added five test files containing seven checks. All seven passed on the server CPU, including extreme-input support-domain coverage.
- Optimized initialization after detecting an impractical large batched least-squares solve at full dimension. Stable direct coefficient initialization reduced full-head construction to 0.302 seconds; curve fitting remains used for adaptive grid preservation.
- Added bounded LayerNorm+tanh input normalization so outlying activations cannot silently fall outside every spline knot interval.
- Added frozen-CLIP feature precomputation in `/dev/shm`, validation-only 30-epoch EMA training, paper/KD protocol-isolated configs, and resumable checkpoints.
- Queued only the seed-3407 paper baseline. It waits for the efficiency diagnosis and three consecutive idle-GPU samples; no three-seed run, test evaluation, search, or new distillation has been started.
