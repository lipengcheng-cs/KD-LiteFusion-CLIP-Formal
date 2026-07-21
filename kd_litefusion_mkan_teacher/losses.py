from typing import Dict, Optional

import torch
import torch.nn.functional as F


def teacher_confidence(teacher_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(teacher_logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(teacher_logits.shape[-1]), device=teacher_logits.device))
    return (1.0 - entropy / max_entropy).clamp(0.0, 1.0)


def _weighted_mean(losses: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return losses.mean()
    weights = weights.to(losses.device).float()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def logits_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"KD temperature must be positive, got {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher logits shape mismatch: {tuple(student_logits.shape)} vs "
            f"{tuple(teacher_logits.shape)}"
        )
    teacher_logits = teacher_logits.detach().to(
        device=student_logits.device,
        dtype=student_logits.dtype,
    )
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise FloatingPointError("Student or teacher logits contain NaN or Inf")
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    if sample_weight is None:
        loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)
    else:
        per_sample = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
        loss = _weighted_mean(per_sample, sample_weight.detach()) * (temperature ** 2)
    if not torch.isfinite(loss):
        raise FloatingPointError("Logits KD loss is NaN or Inf")
    return loss


def feature_kd_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            f"Student/teacher feature shape mismatch: {tuple(student_feature.shape)} vs "
            f"{tuple(teacher_feature.shape)}"
        )
    student_fp32 = student_feature.float()
    teacher_fp32 = teacher_feature.detach().to(device=student_feature.device, dtype=torch.float32)
    if not torch.isfinite(student_fp32).all() or not torch.isfinite(teacher_fp32).all():
        raise FloatingPointError("Student or teacher feature contains NaN or Inf")
    student_norm = F.normalize(student_fp32, dim=-1)
    teacher_norm = F.normalize(teacher_fp32, dim=-1)
    per_sample = 1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=-1)
    loss = _weighted_mean(per_sample, sample_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("Feature KD loss is NaN or Inf")
    return loss


def adapt_student_gate(student_gate: torch.Tensor, teacher_gate: torch.Tensor) -> torch.Tensor:
    if student_gate.shape == teacher_gate.shape:
        return student_gate
    gate_mean = student_gate.mean(dim=-1, keepdim=True)
    if teacher_gate.dim() == 2 and teacher_gate.shape[-1] == 2:
        return torch.cat([1.0 - gate_mean, gate_mean], dim=-1)
    if teacher_gate.dim() == 2 and teacher_gate.shape[-1] == 3:
        fusion_rel = torch.full_like(gate_mean, 0.5)
        return torch.cat([1.0 - gate_mean, gate_mean, fusion_rel], dim=-1)
    return student_gate


def relation_kd_loss(student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    if student_feature.shape[-1] != teacher_feature.shape[-1]:
        raise ValueError(
            "Student/teacher relation feature dimension mismatch: "
            f"{student_feature.shape[-1]} vs {teacher_feature.shape[-1]}"
        )
    student_norm = F.normalize(student_feature, dim=-1)
    teacher_norm = F.normalize(teacher_feature, dim=-1)
    student_relation = student_norm @ student_norm.t()
    teacher_relation = teacher_norm @ teacher_norm.t()
    return F.mse_loss(student_relation, teacher_relation)


def prototype_kd_loss(
    student_feature: torch.Tensor,
    labels: torch.Tensor,
    teacher_prototype: Optional[torch.Tensor],
) -> torch.Tensor:
    if teacher_prototype is None:
        raise ValueError("Prototype KD is enabled but teacher_prototype is missing")
    teacher_prototype = teacher_prototype.to(student_feature.device)
    if student_feature.shape[-1] != teacher_prototype.shape[-1]:
        raise ValueError(
            "Student/teacher prototype dimension mismatch: "
            f"{student_feature.shape[-1]} vs {teacher_prototype.shape[-1]}"
        )
    losses = []
    for cls_id in labels.unique():
        mask = labels == cls_id
        cls_index = int(cls_id.item())
        if cls_index >= teacher_prototype.shape[0]:
            raise ValueError(
                f"Teacher prototypes contain {teacher_prototype.shape[0]} classes, "
                f"label id {cls_index} was requested"
            )
        student_proto = student_feature[mask].mean(dim=0)
        losses.append(F.mse_loss(student_proto, teacher_prototype[cls_index]))
    if not losses:
        raise ValueError("Prototype KD is enabled but the batch contains no labels")
    return torch.stack(losses).mean()


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    teacher_global_prototypes: Optional[torch.Tensor],
    temperature: float,
    weights: Dict[str, float],
    class_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    confidence_weighted_kd: bool = False,
) -> Dict[str, torch.Tensor]:
    logits = outputs["logits"]
    labels = batch["labels"]
    losses = {
        "ce": F.cross_entropy(
            logits,
            labels,
            weight=class_weight.to(logits.device) if class_weight is not None else None,
            label_smoothing=label_smoothing,
        )
    }
    sample_weight = None
    if confidence_weighted_kd:
        if "teacher_logits" not in batch:
            sample_id = batch.get("sample_id", ["UNKNOWN"])[0]
            raise ValueError(
                f"confidence_weighted_kd requires teacher_logits for sample_id: {sample_id}"
            )
        sample_weight = teacher_confidence(batch["teacher_logits"]).detach()

    if weights.get("logits", 0.0) > 0:
        if "teacher_logits" not in batch:
            sample_id = batch.get("sample_id", ["UNKNOWN"])[0]
            raise ValueError(f"Missing teacher logits for sample_id: {sample_id}")
        losses["logits_kd"] = logits_kd_loss(logits, batch["teacher_logits"], temperature, sample_weight)
    if weights.get("feature", 0.0) > 0:
        if "teacher_feature" not in batch:
            sample_id = batch.get("sample_id", ["UNKNOWN"])[0]
            raise ValueError(f"Missing teacher feature for sample_id: {sample_id}")
        losses["feature_kd"] = feature_kd_loss(outputs["feature"], batch["teacher_feature"], sample_weight)
    if weights.get("gate", 0.0) > 0:
        if "teacher_gate" not in batch:
            sample_id = batch.get("sample_id", ["UNKNOWN"])[0]
            raise ValueError(f"Missing teacher gate for sample_id: {sample_id}")
        student_gate = adapt_student_gate(outputs["gate"], batch["teacher_gate"])
        if student_gate.shape != batch["teacher_gate"].shape:
            raise ValueError(
                f"Student/teacher gate shape mismatch: {tuple(student_gate.shape)} vs "
                f"{tuple(batch['teacher_gate'].shape)}"
            )
        losses["gate_kd"] = F.mse_loss(student_gate, batch["teacher_gate"])
    if weights.get("relation", 0.0) > 0:
        if "teacher_feature" not in batch:
            sample_id = batch.get("sample_id", ["UNKNOWN"])[0]
            raise ValueError(f"Missing teacher relation feature for sample_id: {sample_id}")
        losses["relation_kd"] = relation_kd_loss(outputs["feature"], batch["teacher_feature"])
    if weights.get("prototype", 0.0) > 0:
        if "teacher_prototype" in batch:
            if batch["teacher_prototype"].shape != outputs["feature"].shape:
                raise ValueError(
                    "Student/teacher per-sample prototype shape mismatch: "
                    f"{tuple(outputs['feature'].shape)} vs "
                    f"{tuple(batch['teacher_prototype'].shape)}"
                )
            losses["prototype_kd"] = F.mse_loss(
                outputs["feature"], batch["teacher_prototype"].to(outputs["feature"].device)
            )
        else:
            losses["prototype_kd"] = prototype_kd_loss(outputs["feature"], labels, teacher_global_prototypes)

    total = losses["ce"]
    total = total + weights.get("logits", 0.0) * losses.get("logits_kd", logits.new_tensor(0.0))
    total = total + weights.get("feature", 0.0) * losses.get("feature_kd", logits.new_tensor(0.0))
    total = total + weights.get("gate", 0.0) * losses.get("gate_kd", logits.new_tensor(0.0))
    total = total + weights.get("relation", 0.0) * losses.get("relation_kd", logits.new_tensor(0.0))
    total = total + weights.get("prototype", 0.0) * losses.get("prototype_kd", logits.new_tensor(0.0))
    losses["total"] = total
    if not torch.isfinite(total):
        raise FloatingPointError("Total training loss is NaN or Inf")
    return losses
