#!/usr/bin/env python3
"""Precompute frozen OpenAI CLIP token/global features into /dev/shm memmaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


FILES = {"train": "task02_train.tsv", "val": "task02_dev.tsv", "test": "task02_test.tsv"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=FILES, default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def encode(model, images, tokens):
    visual = model.visual
    dtype = model.dtype
    x = visual.conv1(images.to(dtype=dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([class_token, x], dim=1)
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    vision_tokens = x @ visual.proj if visual.proj is not None else x
    vision_global = vision_tokens[:, 0]

    text = model.token_embedding(tokens).to(dtype)
    text = text + model.positional_embedding.to(dtype)
    text = model.transformer(text.permute(1, 0, 2)).permute(1, 0, 2)
    text = model.ln_final(text).to(dtype)
    text_tokens = text @ model.text_projection
    positions = tokens.argmax(dim=-1)
    text_global = text_tokens[torch.arange(text_tokens.shape[0], device=tokens.device), positions]
    outputs = (vision_tokens, text_tokens, vision_global, text_global)
    if not all(torch.isfinite(value).all() for value in outputs):
        raise FloatingPointError("non-finite CLIP feature")
    return outputs


def open_arrays(split_dir: Path, count: int):
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "vision_tokens": np.lib.format.open_memmap(
            split_dir / "vision_tokens.npy", mode="w+", dtype=np.float16, shape=(count, 577, 768)
        ),
        "text_tokens": np.lib.format.open_memmap(
            split_dir / "text_tokens.npy", mode="w+", dtype=np.float16, shape=(count, 77, 768)
        ),
        "vision_global": np.lib.format.open_memmap(
            split_dir / "vision_global.npy", mode="w+", dtype=np.float16, shape=(count, 768)
        ),
        "text_global": np.lib.format.open_memmap(
            split_dir / "text_global.npy", mode="w+", dtype=np.float16, shape=(count, 768)
        ),
        "labels": np.lib.format.open_memmap(
            split_dir / "labels.npy", mode="w+", dtype=np.int64, shape=(count,)
        ),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CLIP feature extraction")
    data_dir, image_root, cache_root, clip_checkpoint = (
        args.data_dir.resolve(), args.image_root.resolve(), args.cache_root.resolve(), args.clip_checkpoint.resolve()
    )
    if not str(cache_root).startswith("/dev/shm/"):
        raise ValueError("paper feature cache must be placed under /dev/shm to protect root disk")
    cache_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, preprocess = clip.load(str(clip_checkpoint), device=device, jit=False)
    model.float().eval().requires_grad_(False)
    clip_hash = sha256(clip_checkpoint)

    for split in args.splits:
        frame = pd.read_csv(data_dir / FILES[split], sep="\t")
        split_dir = cache_root / split
        metadata_path = split_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            required = [split_dir / f"{name}.npy" for name in ("vision_tokens", "text_tokens", "vision_global", "text_global", "labels")]
            if metadata.get("status") == "PASS" and metadata.get("count") == len(frame) and all(path.is_file() for path in required):
                print(f"SKIP_COMPLETE {split} {split_dir}")
                continue
        arrays = open_arrays(split_dir, len(frame))
        sample_ids = []
        for start in tqdm(range(0, len(frame), args.batch_size), desc=f"CLIP cache {split}"):
            end = min(start + args.batch_size, len(frame))
            block = frame.iloc[start:end]
            image_tensors = []
            for relative in block["image_path"].astype(str):
                path = image_root / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as image:
                    image_tensors.append(preprocess(image.convert("RGB")))
            images = torch.stack(image_tensors).to(device, non_blocking=False)
            tokens = clip.tokenize(block["tweet_text"].astype(str).tolist(), truncate=True).to(device)
            features = encode(model, images, tokens)
            for name, value in zip(("vision_tokens", "text_tokens", "vision_global", "text_global"), features):
                arrays[name][start:end] = value.detach().cpu().numpy().astype(np.float16, copy=False)
            arrays["labels"][start:end] = block["label_id"].to_numpy(dtype=np.int64)
            sample_ids.extend(block["image_id"].astype(str).tolist())
        for value in arrays.values():
            value.flush()
        (split_dir / "sample_ids.json").write_text(json.dumps(sample_ids), encoding="utf-8")
        metadata = {
            "status": "PASS",
            "split": split,
            "count": len(frame),
            "unique_sample_ids": len(set(sample_ids)),
            "tsv": str(data_dir / FILES[split]),
            "tsv_sha256": sha256(data_dir / FILES[split]),
            "clip_checkpoint": str(clip_checkpoint),
            "clip_sha256": clip_hash,
            "clip_precision": "FP32 encode; FP16 memmap storage",
            "shapes": {name: list(value.shape) for name, value in arrays.items()},
            "label_order": {
                "affected_individuals": 0,
                "infrastructure_and_utility_damage": 1,
                "not_humanitarian": 2,
                "other_relevant_information": 3,
                "rescue_volunteering_or_donation_effort": 4,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"status": "PASS", "split": split, "count": len(frame)}))


if __name__ == "__main__":
    main()
