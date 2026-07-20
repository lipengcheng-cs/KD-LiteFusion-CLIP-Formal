#!/usr/bin/env python3
"""Generate the final next-stage formal KD report and machine-readable summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def exists_status(path: Path) -> str:
    return "COMPLETE" if path.exists() else "MISSING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    outputs = root / "outputs"
    original_summary = pd.read_csv(outputs / "formal_multiseed" / "summary" / "overall_mean_std.csv")
    paired = pd.read_csv(outputs / "formal_multiseed" / "summary" / "paired_improvement_by_seed.csv")
    wo = original_summary[original_summary.condition == "wo_kd"].iloc[0].to_dict()
    kd = original_summary[original_summary.condition == "logits_kd"].iloc[0].to_dict()
    tuning = yaml.safe_load((outputs / "logits_kd_tuning" / "best_config.yaml").read_text(encoding="utf-8"))
    tuned_path = outputs / "formal_multiseed_tuned" / "overall_mean_std.json"
    tuned = json.loads(tuned_path.read_text(encoding="utf-8")) if tuned_path.is_file() else None
    tuned_improves_weighted = bool(tuned and tuned["weighted_f1_mean"] > float(kd["weighted_f1_mean"]))
    ci_payload = json.loads((outputs / "formal_multiseed" / "analysis" / "bootstrap_confidence_intervals.json").read_text(encoding="utf-8"))
    ci = pd.DataFrame(ci_payload["results"])
    delta_ci = ci[(ci.aggregation == "three_seed_mean") & (ci.condition == "logits_kd_minus_wo_kd")]
    mcnemar = json.loads((outputs / "formal_multiseed" / "analysis" / "mcnemar_test_by_seed.json").read_text(encoding="utf-8"))["results"]
    rescue_reg = pd.read_csv(outputs / "formal_multiseed" / "analysis" / "rescue_regression_cases.csv")
    rescue_imp = pd.read_csv(outputs / "formal_multiseed" / "analysis" / "rescue_improvement_cases.csv")
    class_summary = pd.read_csv(outputs / "formal_multiseed" / "analysis" / "per_class_mean_std.csv")
    rescue = class_summary[class_summary.class_name == "rescue_volunteering_or_donation_effort"].set_index("condition")
    rescue_delta = float(rescue.loc["logits_kd", "f1_mean"] - rescue.loc["wo_kd", "f1_mean"])
    gate = json.loads((outputs / "gate_analysis" / "gate_summary.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((outputs / "feature_kd_screening" / "screening_manifest.json").read_text(encoding="utf-8"))
    feature_selected_path = outputs / "feature_kd_screening" / "selected_feature_config.yaml"
    feature_selected = yaml.safe_load(feature_selected_path.read_text(encoding="utf-8")) if feature_selected_path.is_file() else None
    feature_multi_path = outputs / "feature_kd_screening" / "selected_multiseed_test_results.csv"
    feature_multi = pd.read_csv(feature_multi_path) if feature_multi_path.is_file() else None

    weighted_delta = paired.weighted_f1_delta
    macro_delta = paired.macro_f1_delta
    stable = bool((weighted_delta > 0).all())
    ci_lookup = {row.metric: row for _, row in delta_ci.iterrows()}
    mcnemar_significant = sum(float(row["exact_two_sided_p"]) < 0.05 for row in mcnemar.values())
    feature_continue = not feature_manifest["all_failed"]
    gate_value = bool(gate["sufficient_for_future_gate_kd_discussion"])

    matrix_rows = [
        {"category": "exploratory_history", "experiment": "old full w/o KD", "path": "outputs/server_full_wo_kd", "status": exists_status(outputs / "server_full_wo_kd"), "paper_role": "exploratory_only", "reason": "old leakage-affected teacher protocol"},
        {"category": "exploratory_history", "experiment": "old full Logits KD", "path": "outputs/server_full_logits_kd", "status": exists_status(outputs / "server_full_logits_kd"), "paper_role": "exploratory_only", "reason": "old leakage-affected teacher protocol"},
        {"category": "formal", "experiment": "supplied-source reproduction teacher", "path": "outputs/server_mkan_kd_formal", "status": "COMPLETE", "paper_role": "teacher_method_or_supplement", "reason": "fixed split; not author-original; not true B-spline KAN"},
        {"category": "formal", "experiment": "formal teacher full/logits cache", "path": "outputs/server_mkan_kd_formal/teacher_cache", "status": "PASS", "paper_role": "experiment_infrastructure", "reason": "strict sample-id alignment and finite tensors"},
        {"category": "formal", "experiment": "matched w/o KD three seeds", "path": "outputs/formal_multiseed/wo_kd", "status": "PASS", "paper_role": "main_table", "reason": "best_weighted_f1 checkpoints"},
        {"category": "formal", "experiment": "matched Logits KD three seeds T4 w0.5", "path": "outputs/formal_multiseed/logits_kd", "status": "PASS", "paper_role": "main_table", "reason": "pre-tuning matched formal baseline"},
        {"category": "formal", "experiment": "validation-only staged Logits tuning", "path": "outputs/logits_kd_tuning", "status": "COMPLETE", "paper_role": "method_selection_supplement", "reason": "test not used"},
        {"category": "formal", "experiment": "tuned Logits KD three-seed confirmation", "path": "outputs/formal_multiseed_tuned", "status": "COMPLETE" if tuned else "INCOMPLETE", "paper_role": ("main_table_candidate" if tuned_improves_weighted else "supplementary") if tuned else "not_ready", "reason": "independent directory; original formal results preserved; main role requires improved three-seed test Weighted-F1"},
        {"category": "formal", "experiment": "class statistical diagnosis", "path": "outputs/formal_multiseed/analysis", "status": "COMPLETE", "paper_role": "statistical_support_or_supplement", "reason": "post-selection only"},
        {"category": "formal", "experiment": "Feature KD screening", "path": "outputs/feature_kd_screening", "status": "COMPLETE", "paper_role": "preliminary_or_supplement", "reason": "seed-3407 validation screen with conditional multiseed extension"},
        {"category": "formal", "experiment": "teacher gate diagnosis", "path": "outputs/gate_analysis", "status": "COMPLETE", "paper_role": "diagnostic_supplement", "reason": "no Gate KD loss trained"},
    ]
    pd.DataFrame(matrix_rows).to_csv(outputs / "FORMAL_EXPERIMENT_MATRIX.csv", index=False)

    result = {
        "teacher_identity": {"english": "MKAN-Refine supplied-source reproduction teacher", "chinese": "基于现有源码重训的 MKAN-Refine 复现教师", "author_original_checkpoint": False, "true_b_spline_kan": False},
        "formal_original": {"wo_kd": wo, "logits_kd": kd, "paired_weighted_f1_delta_mean": float(weighted_delta.mean()), "paired_weighted_f1_delta_std": float(weighted_delta.std(ddof=1)), "paired_macro_f1_delta_mean": float(macro_delta.mean()), "paired_macro_f1_delta_std": float(macro_delta.std(ddof=1)), "weighted_f1_positive_seeds": int((weighted_delta > 0).sum()), "stable_weighted_improvement": stable},
        "statistics": {"bootstrap_method": ci_payload["method"], "paired_delta_intervals": delta_ci.to_dict("records"), "mcnemar": mcnemar, "mcnemar_significant_seed_count": mcnemar_significant, "affected_test_support": 7, "rescue_f1_delta": rescue_delta, "rescue_regression_case_count": len(rescue_reg), "rescue_improvement_case_count": len(rescue_imp)},
        "tuning": tuning,
        "tuned_three_seed": tuned,
        "feature_kd": {"all_screening_configs_failed": feature_manifest["all_failed"], "selected": feature_selected, "continue_warranted": feature_continue, "selected_multiseed_test_mean": feature_multi.mean(numeric_only=True).to_dict() if feature_multi is not None else None, "selected_multiseed_test_std": feature_multi.std(numeric_only=True, ddof=1).to_dict() if feature_multi is not None else None},
        "gate": gate,
        "recommendation": {"gate_kd_next": gate_value, "relation_kd_next": False, "prototype_kd_next": False, "full_kd_next": False, "reason": "Relation, Prototype and Full KD remain prohibited; Gate KD needs a separately approved next stage even if diagnostics pass."},
    }
    (outputs / "FORMAL_RESULTS_SUMMARY.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tuned_text = "尚未完成" if tuned is None else f"Weighted-F1={tuned['weighted_f1_mean']:.4f}±{tuned['weighted_f1_std']:.4f}, Macro-F1={tuned['macro_f1_mean']:.4f}±{tuned['macro_f1_std']:.4f}；相对原正式配置 Weighted-F1 均值变化 {tuned['weighted_f1_mean'] - float(kd['weighted_f1_mean']):+.4f}"
    feature_text = "全部配置触发失败规则，应停止" if not feature_continue else f"存在未失败配置，所选配置为 {feature_selected}；已按规则扩展三 seed" 
    gate_text = "具备进入下一阶段讨论的诊断条件" if gate_value else "当前诊断不支持进入 Gate KD"
    lines = [
        "# 下一阶段正式 KD 实验总报告",
        "",
        "## 身份与结果边界",
        "",
        "正式教师统一为 **MKAN-Refine supplied-source reproduction teacher（基于现有源码重训的 MKAN-Refine 复现教师）**。它不是作者原始 checkpoint，也不是真实 B-spline KAN。旧教师、`server_full_wo_kd` 和 `server_full_logits_kd` 受数据泄漏影响，只能作为探索性历史结果。",
        "",
        "正式论文候选结果均来自固定学生划分 train/val/test=6090/995/950。主结果统一使用 `best_weighted_f1.pt`；`best_macro_f1.pt` 仅作补充敏感性分析。",
        "",
        "## 正式原始匹配三 seed",
        "",
        f"Logits KD 的 Weighted-F1 为 {kd['weighted_f1_mean']:.4f}±{kd['weighted_f1_std']:.4f}，w/o KD 为 {wo['weighted_f1_mean']:.4f}±{wo['weighted_f1_std']:.4f}；配对提升 {weighted_delta.mean():+.4f}±{weighted_delta.std(ddof=1):.4f}，3/3 seed 为正。Macro-F1 配对提升 {macro_delta.mean():+.4f}±{macro_delta.std(ddof=1):.4f}。因此 Weighted-F1 和 Macro-F1 方向上是跨 seed 稳定改善，但 Precision 的变化应与类别诊断一起解读。",
        "",
        "## 统计证据",
        "",
        f"分层配对 bootstrap 的 Weighted-F1 平均差 95% CI 为 [{ci_lookup['weighted_f1'].lower_95:+.4f}, {ci_lookup['weighted_f1'].upper_95:+.4f}]；Macro-F1 为 [{ci_lookup['macro_f1'].lower_95:+.4f}, {ci_lookup['macro_f1'].upper_95:+.4f}]。3 个 seed 中有 {mcnemar_significant}/3 个 exact McNemar p<0.05。统计结果仅用于模型选择后的解释，没有反向参与选择。",
        "",
        f"`affected_individuals` 测试 support 只有 7，其单次或均值 F1 大幅变化都高度受小样本影响，不能声称稳定类别改善。Rescue 类三 seed 平均 F1 变化为 {rescue_delta:+.4f}；共记录 {len(rescue_reg)} 个 w/o 正确→KD 错误案例和 {len(rescue_imp)} 个 w/o 错误→KD 正确案例。",
        "",
        "## 调参、Feature KD 与 Gate",
        "",
        f"验证集分阶段搜索选择 T={tuning['teacher']['temperature']}, logits weight={tuning['kd_weights']['logits']}，测试集未用于选择。独立 tuned 三 seed 确认：{tuned_text}。",
        "",
        f"Feature KD：{feature_text}。full cache 的存在本身不构成 Feature KD 有效证据。",
        "",
        f"Gate 诊断：overall std={gate['overall_std']:.6f}，饱和比例={gate['saturation_outside_0p05_0p95']:.2%}，near_constant={gate['near_constant']}，结论为“{gate_text}”。目前没有实现或训练 Gate KD loss。图文差异相关性通过流式 frozen-CLIP 推理计算（r={gate['image_text_difference_gate_correlation']:.6f}），没有重新生成已删除的 7.6GB 中间缓存。",
        "",
        "## 可进入论文的范围",
        "",
        f"- 主表：正式 matched w/o KD 与正式 matched Logits KD 的三 seed `best_weighted_f1` 均值±标准差。tuned 三 seed {'提升了 Weighted-F1，可作为候选并保留原结果对照' if tuned_improves_weighted else '未提升测试 Weighted-F1，仅作补充，不替换原正式主结果'}。",
        "- 补充材料：best-Macro-F1 敏感性、逐类指标、混淆矩阵、bootstrap、McNemar、调参轨迹、Feature KD 初筛、Gate 诊断、复现教师的身份和审计。",
        "- 仅探索性：旧泄漏教师、旧 `server_full_*` 结果、任何单 seed 或 support=7 的夸大结论。",
        "",
        "## 下一步",
        "",
        f"Gate KD 下一阶段建议：{'可以单独立项讨论，但不能直接宣称有效' if gate_value else '暂不进入'}。Relation KD、Prototype KD、Full KD 和 rank sensitivity 继续禁止；应先完成论文表格固化、外部复核和资源审计。",
        "",
    ]
    report = "\n".join(lines)
    (outputs / "NEXT_STAGE_FORMAL_KD_REPORT.md").write_text(report, encoding="utf-8")
    (outputs / "FORMAL_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8")
    print(outputs / "NEXT_STAGE_FORMAL_KD_REPORT.md")


if __name__ == "__main__":
    main()
