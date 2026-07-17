# Remaining formal experiments changelog

## 2026-07-17 — fair MKAN/LiteFusion efficiency benchmark

### Scope

- Expanded the formal efficiency comparison to the validation-selected best single-seed MKAN reproduction teacher, the formal three-checkpoint MKAN ensemble, LiteFusion-CLIP w/o KD, LiteFusion-CLIP + Logits KD, and any later Feature-KD student that passes screening and completes three seeds.
- The teacher name is fixed to `MKAN-Refine supplied-source reproduction teacher`; the benchmark and report explicitly state that this is neither an author-original checkpoint nor a strict B-spline KAN reproduction.
- The formal ensemble shares one frozen OpenAI CLIP encoder but executes all three MKAN heads, including the zero-weight member, and sums all three checkpoint sizes.

### Files

- Replaced: `efficiency.py`
- Added: `scripts/36_run_efficiency_benchmark.sh`
- Added: `scripts/summarize_efficiency.py`
- Added: `CODE_CHANGELOG_REMAINING_EXPERIMENTS.md`

### Protocol

- V100, OpenAI CLIP ViT-L/14@336px, 336×336 images, 77 tokens, FP32, `model.eval()`, `torch.inference_mode()`.
- Batch sizes 1 and 8; 30 warm-up iterations; 100 timed iterations; three rounds; CUDA synchronization around every timed call.
- End-to-end: image tensor/token ids → CLIP → Fusion/Gate/Classifier → logits.
- Fusion/Head-only: precomputed CLIP features → complete Fusion/Gate/Classifier → logits.
- w/o KD, Logits KD, and a selected later KD student share one strict runtime measurement because they have the same inference graph and do not load a teacher.
- FLOPs use `torch.profiler(with_flops=True)`; MACs are reported at 2 FLOPs per MAC, with profiler coverage limitations disclosed.

### Commands and task

- Validation: `python -m compileall -q .`
- Shell validation: `bash -n scripts/36_run_efficiency_benchmark.sh`
- Formal run: `scripts/36_run_efficiency_benchmark.sh`
- tmux: `efficiency_20260717`
- Log: `logs/efficiency/formal_efficiency.log`
- Output: `outputs/efficiency/`

### Errors and fixes

- `matplotlib` was absent from the `kdclip` environment and the package download was too slow. The incomplete download was stopped before installation. Plot generation was changed to the already-installed Pillow package; all four PNGs are written with 300-DPI metadata and paired CSV data, avoiding a new environment dependency.
- Formal timing is intentionally queued behind the active Feature-KD and finalization sessions to prevent GPU contention.

### Git

- Commit: pending successful benchmark and output validation.
