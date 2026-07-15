# Full-data initial run (w/o KD)

This folder runs the initial CrisisMMD Task2 LiteFusion-CLIP experiment without knowledge distillation.

Run on the server in this order:

```bash
bash scripts/01_check_env.sh
bash scripts/02_check_data.sh
bash scripts/03_train_full_wo_kd.sh
bash scripts/04_evaluate_full_wo_kd.sh
```

- Step 1 succeeds when CUDA/GPU information is printed and `local CLIP offline load: OK` appears.
- Step 2 succeeds when split/label counts, checked image paths, and `full data check: OK` appear.
- Step 3 succeeds when `outputs/full_wo_kd/best.pt` is created.
- Step 4 succeeds when `eval_metrics.json` and `test_predictions.csv` are created.

On failure, send the matching file from `logs/` to GPT: `check_env.log`, `check_data.log`, `full_wo_kd.log`, or `evaluate_full_wo_kd.log`.
