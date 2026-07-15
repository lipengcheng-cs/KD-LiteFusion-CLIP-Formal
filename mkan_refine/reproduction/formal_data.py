"""Fixed-split datasets for the formal supplied-source MKAN reproduction teacher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


NATIVE_LABEL_TO_ID = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "not_humanitarian": 2,
    "other_relevant_information": 3,
    "rescue_volunteering_or_donation_effort": 4,
}
STUDENT_LABEL_TO_ID = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "rescue_volunteering_or_donation_effort": 2,
    "other_relevant_information": 3,
    "not_humanitarian": 4,
}
LABEL_MERGE = {
    "injured_or_dead_people": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "vehicle_damage": "infrastructure_and_utility_damage",
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id_hash(ids) -> str:
    payload = "\n".join(str(value) for value in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_label(value: object) -> str:
    label = str(value).strip()
    label = LABEL_MERGE.get(label, label)
    if label not in STUDENT_LABEL_TO_ID:
        raise ValueError(f"Unsupported label: {value!r}")
    return label


def load_formal_csv(csv_path: str, expected_counts: Dict[str, int]) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"sample_id", "image_path", "text", "label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["split"].astype(str).str.lower()
    if df["sample_id"].duplicated().any():
        duplicate = df.loc[df["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"Duplicate sample_id: {duplicate}")
    actual = {split: int((df["split"] == split).sum()) for split in ("train", "val", "test")}
    if actual != {key: int(value) for key, value in expected_counts.items()}:
        raise ValueError(f"Fixed split count mismatch: actual={actual}, expected={expected_counts}")
    if set(df["split"]) != {"train", "val", "test"}:
        raise ValueError(f"Unexpected split values: {sorted(set(df['split']))}")
    df["canonical_label"] = df["label"].map(canonical_label)
    df["native_label_id"] = df["canonical_label"].map(NATIVE_LABEL_TO_ID)
    df["student_label_id"] = df["canonical_label"].map(STUDENT_LABEL_TO_ID)
    return df


def build_transform(image_size: int = 336):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class RawFormalDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: str, tokenizer: Callable):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = build_transform(336)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]
        image_path = str(row["image_path"])
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)
        try:
            with Image.open(image_path) as image:
                pixel_values = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(
                f"Image load failed for sample_id={row['sample_id']}, path={image_path}: {exc}"
            ) from exc
        return {
            "sample_id": str(row["sample_id"]),
            "pixel_values": pixel_values,
            "input_ids": self.tokenizer([str(row["text"])], truncate=True).squeeze(0),
            "label": torch.tensor(int(row["native_label_id"]), dtype=torch.long),
        }


class CachedFeatureDataset(Dataset):
    REQUIRED = (
        "sample_ids", "labels", "vision_tokens", "text_tokens", "vision_global", "text_global"
    )

    def __init__(self, payload: Dict):
        missing = [key for key in self.REQUIRED if key not in payload]
        if missing:
            raise ValueError(f"Feature cache missing: {missing}")
        length = len(payload["sample_ids"])
        if any(len(payload[key]) != length for key in self.REQUIRED[1:]):
            raise ValueError("Feature cache fields have inconsistent lengths")
        self.payload = payload

    def __len__(self) -> int:
        return len(self.payload["sample_ids"])

    def __getitem__(self, index: int) -> Dict:
        return {
            "sample_id": str(self.payload["sample_ids"][index]),
            "label": self.payload["labels"][index].long(),
            "vision_tokens": self.payload["vision_tokens"][index],
            "text_tokens": self.payload["text_tokens"][index],
            "vision_global": self.payload["vision_global"][index],
            "text_global": self.payload["text_global"][index],
        }
