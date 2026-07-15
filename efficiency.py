import argparse
import json
import os
import time

import torch

from kd_litefusion_mkan_teacher.model import KDLiteFusionCLIP
from kd_litefusion_mkan_teacher.utils import load_checkpoint_state


def parse_args():
    parser = argparse.ArgumentParser(description="Measure KD-LiteFusion-CLIP efficiency")
    parser.add_argument("--clip_model_path", default="/home/lpc/.cache/clip/ViT-L-14-336px.pt")
    parser.add_argument("--ckpt", "--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--seq_len", type=int, default=77)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def module_params(module):
    return sum(p.numel() for p in module.parameters())


def sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model_path = args.clip_model_path
    rank = args.rank
    num_classes = args.num_classes
    if args.ckpt:
        checkpoint = load_checkpoint_state(args.ckpt, device)
        train_args = checkpoint.get("args", {})
        clip_model_path = train_args.get("clip_model_path", clip_model_path)
        rank = int(train_args.get("rank", rank))
        num_classes = len(checkpoint.get("label_to_id", {})) or num_classes

    model = KDLiteFusionCLIP(
        clip_model_path, num_classes, rank=rank, freeze_clip=True, device=device
    ).to(device).eval()
    if args.ckpt:
        model.load_student_state_dict(checkpoint["student_state_dict"])
    total, trainable = count_params(model)

    text_tokens = torch.ones(args.batch_size, args.seq_len, dtype=torch.long, device=device)
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    for _ in range(args.warmup):
        model(text_tokens, images)
    sync_if_needed()

    start = time.perf_counter()
    for _ in range(args.steps):
        model(text_tokens, images)
    sync_if_needed()
    elapsed = time.perf_counter() - start
    end_to_end_latency_ms = elapsed / args.steps * 1000.0
    throughput = args.batch_size * args.steps / elapsed

    vision_feat = torch.randn(args.batch_size, model.dim, device=device)
    text_feat = torch.randn(args.batch_size, model.dim, device=device)
    fusion_feat = model.fusion(vision_feat, text_feat)
    for _ in range(args.warmup):
        model.fusion(vision_feat, text_feat)
    sync_if_needed()
    start = time.perf_counter()
    for _ in range(args.steps * 10):
        model.fusion(vision_feat, text_feat)
    sync_if_needed()
    fusion_elapsed = time.perf_counter() - start
    fusion_latency_ms = fusion_elapsed / (args.steps * 10) * 1000.0

    report = {
        "rank": rank,
        "total_params": total,
        "trainable_params": trainable,
        "clip_params": module_params(model.clip),
        "fusion_params": module_params(model.fusion),
        "gate_params": module_params(model.gate),
        "classifier_params": module_params(model.classifier),
        "fusion_latency_ms": fusion_latency_ms,
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "throughput": throughput,
    }

    try:
        from thop import profile

        flops, _ = profile(model, inputs=(text_tokens, images), verbose=False)
        report["flops"] = int(flops)
    except Exception as exc:
        report["flops"] = None
        report["flops_error"] = str(exc)

    print(json.dumps(report, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"efficiency report saved to {args.output}")


if __name__ == "__main__":
    main()
