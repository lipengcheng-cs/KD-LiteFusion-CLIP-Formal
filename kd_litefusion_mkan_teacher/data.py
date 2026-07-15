import os
import numbers
from typing import Callable, Dict, List, Optional, Tuple

import clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .teacher_cache import TEACHER_KEYS, TeacherCache

REQUIRED_COLUMNS = ("sample_id", "image_path", "text", "label", "split")

# This order is part of the experiment contract.  Do not sort these labels.
CANONICAL_LABELS = (
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian",
)
LABEL_TO_ID = {label: index for index, label in enumerate(CANONICAL_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
LABEL_MERGE = {
    "injured_or_dead_people": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "vehicle_damage": "infrastructure_and_utility_damage",
}


def read_crisismmd_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        duplicate = df.loc[df["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"Dataset contains duplicate sample_id: {duplicate}")
    return df


def build_label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    for value in df["label"].dropna().unique():
        canonical_label(value)
    return dict(LABEL_TO_ID), dict(ID_TO_LABEL)


def canonical_label(value) -> str:
    if isinstance(value, numbers.Integral):
        label_id = int(value)
        if label_id not in ID_TO_LABEL:
            raise ValueError(f"Unknown numeric label id: {label_id}")
        return ID_TO_LABEL[label_id]
    name = LABEL_MERGE.get(str(value), str(value))
    if name not in LABEL_TO_ID:
        raise ValueError(f"Unknown CrisisMMD label: {value!r}")
    return name


class CrisisMMDTask2Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        preprocess: Callable,
        label_to_id: Dict[str, int],
        teacher_cache: Optional[TeacherCache] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.preprocess = preprocess
        self.label_to_id = label_to_id
        self.teacher_cache = teacher_cache

    def __len__(self) -> int:
        return len(self.df)

    def _label_id(self, value) -> int:
        return self.label_to_id[canonical_label(value)]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = str(row["sample_id"])
        img_path = str(row["image_path"])
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        try:
            with Image.open(img_path) as image:
                image_tensor = self.preprocess(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image for sample_id={sample_id!r}, image_path={img_path!r}: {exc}"
            ) from exc

        text_tokens = clip.tokenize(str(row["text"]), truncate=True).squeeze(0)

        item = {
            "sample_id": sample_id,
            "text_tokens": text_tokens,
            "images": image_tensor,
            "labels": torch.tensor(self._label_id(row["label"]), dtype=torch.long),
        }

        if self.teacher_cache is not None:
            cached = self.teacher_cache.get_required(sample_id)
            for key in TEACHER_KEYS:
                if key in cached:
                    item[f"teacher_{key}"] = cached[key].float()

        return item


def collate_batch(samples: List[Dict]) -> Dict:
    out = {
        "sample_id": [x["sample_id"] for x in samples],
        "text_tokens": torch.stack([x["text_tokens"] for x in samples]),
        "images": torch.stack([x["images"] for x in samples]),
        "labels": torch.stack([x["labels"] for x in samples]),
    }
    for key in TEACHER_KEYS:
        teacher_key = f"teacher_{key}"
        if all(teacher_key in x for x in samples):
            out[teacher_key] = torch.stack([x[teacher_key] for x in samples])
    return out


def build_dataloaders(
    csv_path: str,
    image_root: str,
    preprocess: Callable,
    batch_size: int,
    num_workers: int,
    teacher_cache_path: Optional[str] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[int, str], Optional[TeacherCache]]:
    df = read_crisismmd_csv(csv_path)
    label_to_id, id_to_label = build_label_mapping(df)
    teacher_cache = TeacherCache(teacher_cache_path) if teacher_cache_path else None
    if teacher_cache is not None:
        train_ids = df.loc[df["split"].astype(str).str.lower() == "train", "sample_id"].tolist()
        missing = teacher_cache.missing_ids(train_ids)
        if missing:
            raise ValueError(f"Missing teacher logits for sample_id: {missing[0]}")
        extra = sorted(set(teacher_cache.by_id) - set(train_ids))
        if extra:
            raise ValueError(f"Teacher cache contains non-train sample_id: {extra[0]}")

    loaders = {}
    for split in ("train", "val", "test"):
        split_df = df[df["split"].astype(str).str.lower() == split]
        if split_df.empty:
            continue
        dataset = CrisisMMDTask2Dataset(
            split_df,
            image_root=image_root,
            preprocess=preprocess,
            label_to_id=label_to_id,
            teacher_cache=teacher_cache if split == "train" else None,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
    return loaders, label_to_id, id_to_label, teacher_cache
