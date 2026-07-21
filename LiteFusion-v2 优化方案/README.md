# LiteFusion-v2 优化方案

简介：原版学生模型优化。

本目录是 GitHub 入口说明，正式实现位于：

- `kd_litefusion_mkan_teacher/litefusion_v2/`
- `configs/litefusion_v2/`
- `scripts/run_litefusion_v2_screening.py`
- `outputs/litefusion_v2/`（运行结果，不覆盖历史输出）

第一轮严格固定 `interaction_rank=32`、`seed=3407`、w/o KD、4 epochs，
只使用 validation 指标筛选，不执行 test evaluation。

五个候选：

1. `v2_a_residual_only`
2. `v2_p_precision`
3. `v2_b_balanced`
4. `v2_c_compact`
5. `v2_g_grouped`

正式 profiling 区分 `head_only`、`gpu_tensor_end_to_end` 与
`deployment_end_to_end`，并分别报告 CLIP-only、fusion、gate、classifier、
full head；不会把 fusion 单模块耗时命名为 head latency。
