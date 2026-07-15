import argparse
import json
import os

import pandas as pd
import torch
from tqdm import tqdm

from kd_litefusion_mkan_teacher.data import ID_TO_LABEL, LABEL_TO_ID, build_dataloaders
from kd_litefusion_mkan_teacher.metrics import compute_metrics, confusion_matrix_df, per_class_metrics_df
from kd_litefusion_mkan_teacher.model import KDLiteFusionCLIP
from kd_litefusion_mkan_teacher.utils import load_checkpoint_state, move_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KD-LiteFusion-CLIP")
    parser.add_argument("--config", default=None)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--output_csv", default="outputs/test_predictions.csv")
    parser.add_argument("--metrics_json", default=None)
    parser.add_argument("--per_class_csv", default=None)
    parser.add_argument("--confusion_csv", default=None)
    parser.add_argument("--clip_model_path", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--max_text_len", type=int, default=77)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def load_model_config(path):
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install pyyaml to use --config") from exc
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("model", {})


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint_state(args.checkpoint, device)
    train_args = checkpoint.get("args", {})
    model_cfg = load_model_config(args.config)
    clip_model_path = (
        args.clip_model_path
        or model_cfg.get("clip_model_path")
        or train_args.get("clip_model_path")
        or "/home/lpc/.cache/clip/ViT-L-14-336px.pt"
    )
    rank = int(train_args.get("rank", 32))
    dropout = float(train_args.get("dropout", 0.2))
    checkpoint_label_to_id = {str(key): int(value) for key, value in checkpoint.get("label_to_id", {}).items()}
    if not checkpoint_label_to_id:
        raise ValueError("Checkpoint is missing label_to_id")
    if checkpoint_label_to_id != LABEL_TO_ID:
        raise ValueError(
            f"Checkpoint label_to_id does not match the fixed five-class mapping: {checkpoint_label_to_id}"
        )
    num_classes = len(LABEL_TO_ID)

    model = KDLiteFusionCLIP(
        clip_model_path,
        num_classes,
        rank=rank,
        dropout=dropout,
        freeze_clip=True,
        device=device,
    ).to(device)

    loaders, label_to_id, id_to_label, _ = build_dataloaders(
        csv_path=args.csv_path,
        image_root=args.image_root,
        preprocess=model.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        teacher_cache_path=None,
    )
    if args.split not in loaders:
        raise ValueError(f"CSV does not contain split == {args.split}")
    if label_to_id != LABEL_TO_ID or id_to_label != ID_TO_LABEL:
        raise ValueError("Dataset label mapping does not match the fixed five-class mapping")

    student_state = checkpoint.get("student_state_dict")
    if student_state is None:
        raise ValueError("Checkpoint is missing student_state_dict; retrain with the OpenAI CLIP version")
    model.load_student_state_dict(student_state)
    model.eval()

    labels, preds, sample_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loaders[args.split], desc=f"evaluate {args.split}"):
            sample_ids.extend(batch["sample_id"])
            batch = move_to_device(batch, device)
            outputs = model(batch["text_tokens"], batch["images"])
            pred = outputs["logits"].argmax(dim=-1)
            labels.extend(batch["labels"].cpu().tolist())
            preds.extend(pred.cpu().tolist())

    metrics = compute_metrics(labels, preds)
    print(json.dumps(metrics, indent=2))

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    metrics_json = args.metrics_json or os.path.join(os.path.dirname(args.output_csv) or ".", "metrics.json")
    os.makedirs(os.path.dirname(metrics_json) or ".", exist_ok=True)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "pred": preds,
            "label_name": [id_to_label.get(int(x), str(x)) for x in labels],
            "pred_name": [id_to_label.get(int(x), str(x)) for x in preds],
        }
    ).to_csv(args.output_csv, index=False)
    print(f"metrics saved to {metrics_json}")
    print(f"predictions saved to {args.output_csv}")

    per_class_csv = args.per_class_csv or os.path.join(os.path.dirname(args.output_csv) or ".", "per_class_metrics.csv")
    confusion_csv = args.confusion_csv or os.path.join(os.path.dirname(args.output_csv) or ".", "confusion_matrix.csv")
    os.makedirs(os.path.dirname(per_class_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(confusion_csv) or ".", exist_ok=True)
    per_class_metrics_df(labels, preds, id_to_label).to_csv(per_class_csv, index=False)
    confusion_matrix_df(labels, preds, id_to_label).to_csv(confusion_csv)
    print(f"per-class metrics saved to {per_class_csv}")
    print(f"confusion matrix saved to {confusion_csv}")


if __name__ == "__main__":
    main()
