import argparse
import json
import os
import sys
from typing import Dict

import torch
from torch.optim import AdamW
from tqdm import tqdm

from kd_litefusion_mkan_teacher.data import build_dataloaders, build_label_mapping, read_crisismmd_csv
from kd_litefusion_mkan_teacher.losses import compute_total_loss
from kd_litefusion_mkan_teacher.metrics import compute_metrics, format_metrics
from kd_litefusion_mkan_teacher.model import KDLiteFusionCLIP
from kd_litefusion_mkan_teacher.utils import atomic_torch_save, ensure_dir, move_to_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train KD-LiteFusion-CLIP")
    parser.add_argument("--config", default=None, help="YAML experiment config. CLI args override config values.")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--teacher_cache", default=None)
    parser.add_argument("--output_dir", default="outputs/kd_litefusion")
    parser.add_argument("--clip_backend", default="openai", choices=("openai",))
    parser.add_argument("--clip_model_name", default="ViT-L/14@336px")
    parser.add_argument("--clip_model_path", default="/home/lpc/.cache/clip/ViT-L-14-336px.pt")
    parser.add_argument("--clip_frozen", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--max_text_len", type=int, default=77)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--logits_kd_weight", "--logits_weight", type=float, default=0.5)
    parser.add_argument("--feature_kd_weight", "--feature_weight", type=float, default=0.0)
    parser.add_argument("--gate_kd_weight", "--gate_weight", type=float, default=0.0)
    parser.add_argument("--relation_kd_weight", "--relation_weight", type=float, default=0.0)
    parser.add_argument("--prototype_kd_weight", "--proto_weight", type=float, default=0.0)
    parser.add_argument("--disable_kd", action="store_true")
    parser.add_argument("--confidence_weighted_kd", action="store_true")
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument(
        "--class_weight_method",
        choices=("inverse_frequency", "inverse_freq", "effective_num", "none"),
        default="inverse_frequency",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--save_best_by", choices=("weighted_f1", "macro_f1", "accuracy"), default="weighted_f1")
    parser.add_argument("--use_kd_schedule", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    apply_config(args, supplied_dests(parser))
    validate_args(args)
    return args


def load_config(path: str) -> Dict:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install pyyaml to use --config") from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def supplied_dests(parser: argparse.ArgumentParser) -> set:
    supplied = set()
    argv = set(sys.argv[1:])
    for action in parser._actions:
        if any(option in argv or any(arg.startswith(f"{option}=") for arg in argv) for option in action.option_strings):
            supplied.add(action.dest)
    return supplied


def apply_config(args, supplied: set) -> None:
    cfg = load_config(args.config)
    if not cfg:
        return
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    train = cfg.get("train", {})
    loss = cfg.get("loss", {})
    teacher = cfg.get("teacher", {})
    weights = cfg.get("kd_weights", {})
    values = {
        "csv_path": data.get("csv_path"),
        "image_root": data.get("image_root"),
        "teacher_cache": data.get("teacher_cache"),
        "output_dir": train.get("output_dir"),
        "clip_backend": model.get("clip_backend"),
        "clip_model_name": model.get("clip_model_name"),
        "clip_model_path": model.get("clip_model_path"),
        "clip_frozen": model.get("clip_frozen"),
        "rank": model.get("rank"),
        "dropout": model.get("dropout"),
        "epochs": train.get("epochs"),
        "batch_size": train.get("batch_size"),
        "num_workers": train.get("num_workers"),
        "lr": train.get("lr"),
        "weight_decay": train.get("weight_decay"),
        "disable_kd": train.get("disable_kd"),
        "save_best_by": train.get("save_best_by"),
        "use_class_weight": loss.get("use_class_weight"),
        "class_weight_method": loss.get("class_weight_method"),
        "label_smoothing": loss.get("label_smoothing"),
        "temperature": teacher.get("temperature"),
        "confidence_weighted_kd": teacher.get("confidence_weighted_kd"),
        "logits_kd_weight": weights.get("logits"),
        "feature_kd_weight": weights.get("feature"),
        "gate_kd_weight": weights.get("gate"),
        "relation_kd_weight": weights.get("relation"),
        "prototype_kd_weight": weights.get("prototype"),
    }
    for attr, value in values.items():
        if value is not None and attr not in supplied:
            setattr(args, attr, value)
    if cfg.get("kd_schedule") and "use_kd_schedule" not in supplied:
        args.use_kd_schedule = True


def validate_args(args) -> None:
    if not args.csv_path or not args.image_root:
        raise ValueError("Provide data.csv_path and data.image_root")
    if not args.clip_frozen:
        raise ValueError("This experiment requires a frozen OpenAI CLIP encoder")
    if args.temperature <= 0:
        raise ValueError("teacher.temperature must be positive")
    unsupported_weights = {
        "gate": args.gate_kd_weight,
        "relation": args.relation_kd_weight,
        "prototype": args.prototype_kd_weight,
    }
    nonzero = {name: value for name, value in unsupported_weights.items() if float(value) != 0.0}
    if nonzero:
        raise ValueError(f"This stage supports Logits KD and Feature KD only; these KD weights must be 0: {nonzero}")
    if args.use_kd_schedule:
        raise ValueError("KD scheduling is not enabled for the minimal Logits KD experiment")
    if args.confidence_weighted_kd:
        raise ValueError("confidence_weighted_kd must be false for standard Logits KD")
    if args.disable_kd:
        if args.teacher_cache:
            raise ValueError("w/o KD must not load a teacher cache")
        if args.logits_kd_weight > 0 or args.feature_kd_weight > 0:
            raise ValueError("w/o KD must set Logits and Feature KD weights to 0")
        return
    if not args.teacher_cache:
        raise ValueError("Logits KD requires data.teacher_cache")
    if not os.path.isfile(args.teacher_cache):
        raise FileNotFoundError(f"Teacher logits cache not found: {args.teacher_cache}")
    if args.logits_kd_weight <= 0 and args.feature_kd_weight <= 0:
        raise ValueError("At least one of kd_weights.logits or kd_weights.feature must be greater than 0")
    if os.path.normpath(args.output_dir) == os.path.normpath("outputs/full_wo_kd"):
        raise ValueError("Logits KD must not write to outputs/full_wo_kd")


def compute_class_weight(csv_path: str, label_to_id: Dict[str, int], method: str) -> torch.Tensor:
    if method == "none":
        return torch.ones(len(label_to_id), dtype=torch.float32)
    df = read_crisismmd_csv(csv_path)
    from kd_litefusion_mkan_teacher.data import canonical_label

    train_labels = df.loc[df["split"].astype(str).str.lower() == "train", "label"].map(canonical_label)
    counts = torch.ones(len(label_to_id), dtype=torch.float32)
    for label, count in train_labels.value_counts().items():
        counts[label_to_id[label]] = float(count)
    if method == "effective_num":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
    else:
        weights = 1.0 / counts
    return weights / weights.mean().clamp_min(1e-8)


def kd_weights(args) -> Dict[str, float]:
    return {
        "logits": 0.0 if args.disable_kd else float(args.logits_kd_weight),
        "feature": 0.0 if args.disable_kd else float(args.feature_kd_weight),
        "gate": 0.0,
        "relation": 0.0,
        "prototype": 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    labels, preds = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(batch["text_tokens"], batch["images"])
        labels.extend(batch["labels"].cpu().tolist())
        preds.extend(outputs["logits"].argmax(dim=-1).cpu().tolist())
    return compute_metrics(labels, preds)


def checkpoint_payload(args, model, epoch, metrics, label_to_id, id_to_label, teacher_cache) -> Dict:
    teacher_checkpoint = None
    if teacher_cache is not None:
        teacher_checkpoint = teacher_cache.metadata.get(
            "teacher_checkpoints", teacher_cache.metadata.get("teacher_checkpoint")
        )
    return {
        "epoch": int(epoch),
        "validation_metrics": dict(metrics),
        "student_state_dict": model.student_state_dict(),
        "label_to_id": dict(label_to_id),
        "id_to_label": dict(id_to_label),
        "args": vars(args).copy(),
        "kd_temperature": float(args.temperature),
        "logits_kd_weight": 0.0 if args.disable_kd else float(args.logits_kd_weight),
        "teacher_cache_path": os.path.abspath(args.teacher_cache) if args.teacher_cache else None,
        "teacher_checkpoint_path": teacher_checkpoint,
        "config_snapshot": load_config(args.config),
    }


def atomic_json_save(payload, path: str) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = read_crisismmd_csv(args.csv_path)
    label_to_id, _ = build_label_mapping(df)
    model = KDLiteFusionCLIP(
        args.clip_model_path,
        len(label_to_id),
        rank=args.rank,
        dropout=args.dropout,
        freeze_clip=args.clip_frozen,
        device=device,
    ).to(device)
    loaders, label_to_id, id_to_label, teacher_cache = build_dataloaders(
        csv_path=args.csv_path,
        image_root=args.image_root,
        preprocess=model.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        teacher_cache_path=args.teacher_cache,
    )
    if "train" not in loaders:
        raise ValueError("CSV must contain split == train")
    val_loader = loaders.get("val", loaders.get("test"))
    if val_loader is None:
        raise ValueError("CSV must contain split == val or split == test")

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    class_weight = None
    if args.use_class_weight:
        class_weight = compute_class_weight(args.csv_path, label_to_id, args.class_weight_method).to(device)
        print(f"class weights: {[round(value, 4) for value in class_weight.cpu().tolist()]}")
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as handle:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, handle, indent=2, ensure_ascii=False)

    best_weighted_key = (-1.0, -1.0)
    best_macro_key = (-1.0, -1.0)
    history = []
    weights = kd_weights(args)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(loaders["train"], desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["text_tokens"], batch["images"])
            losses = compute_total_loss(
                outputs=outputs,
                batch=batch,
                teacher_global_prototypes=None,
                temperature=args.temperature,
                weights=weights,
                class_weight=class_weight,
                label_smoothing=args.label_smoothing,
                confidence_weighted_kd=False,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += losses["total"].item()
            progress.set_postfix(loss=f"{losses['total'].item():.4f}")

        metrics = evaluate(model, val_loader, device)
        average_loss = total_loss / max(1, len(loaders["train"]))
        print(
            f"epoch {epoch} train_loss: {average_loss:.4f} | "
            f"kd_weights: {weights} | {format_metrics(metrics)}"
        )
        payload = checkpoint_payload(args, model, epoch, metrics, label_to_id, id_to_label, teacher_cache)
        history.append({
            "epoch": epoch,
            "train_loss": average_loss,
            "validation_metrics": dict(metrics),
            "kd_weights": dict(weights),
        })
        atomic_json_save(history, os.path.join(args.output_dir, "train_history.json"))
        weighted_key = (metrics["weighted_f1"], metrics["macro_f1"])
        macro_key = (metrics["macro_f1"], metrics["weighted_f1"])
        if weighted_key > best_weighted_key:
            best_weighted_key = weighted_key
            atomic_torch_save(payload, os.path.join(args.output_dir, "best_weighted_f1.pt"))
            atomic_torch_save(payload, os.path.join(args.output_dir, "best.pt"))
            print("saved best weighted-F1 checkpoint")
        if macro_key > best_macro_key:
            best_macro_key = macro_key
            atomic_torch_save(payload, os.path.join(args.output_dir, "best_macro_f1.pt"))
            print("saved best macro-F1 checkpoint")
        atomic_torch_save(payload, os.path.join(args.output_dir, "last.pt"))

    atomic_json_save(
        {
            "config_path": os.path.abspath(args.config) if args.config else None,
            "resolved_args": vars(args).copy(),
            "config": load_config(args.config),
            "primary_checkpoint": "best_weighted_f1.pt",
            "sensitivity_checkpoint": "best_macro_f1.pt",
        },
        os.path.join(args.output_dir, "config_snapshot.json"),
    )


if __name__ == "__main__":
    main()
