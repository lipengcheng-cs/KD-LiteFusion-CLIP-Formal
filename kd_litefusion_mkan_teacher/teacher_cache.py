from typing import Any, Dict, List, Mapping, Optional

import torch

TEACHER_KEYS = ("logits", "feature", "gate", "prototype")
EXPECTED_LABEL_TO_ID = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "rescue_volunteering_or_donation_effort": 2,
    "other_relevant_information": 3,
    "not_humanitarian": 4,
}


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_label_mapping(mapping: Mapping) -> Dict[str, int]:
    return {str(key): int(value) for key, value in mapping.items()}


class TeacherCache:
    """Strict, sample-id keyed logits cache for the train split."""

    def __init__(self, path: Optional[str] = None):
        self.by_id: Dict[str, Dict[str, torch.Tensor]] = {}
        self.global_prototypes: Optional[torch.Tensor] = None
        self.metadata: Dict[str, Any] = {}
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        raw = _torch_load(path)
        if not isinstance(raw, Mapping):
            raise ValueError("Teacher cache must be a dictionary")
        required = {"sample_ids", "logits", "label_to_id", "id_to_label", "split"}
        missing_fields = sorted(required - set(raw))
        if missing_fields:
            raise ValueError(f"Teacher cache is missing fields: {missing_fields}")

        sample_ids = [str(value) for value in raw["sample_ids"]]
        if len(sample_ids) != len(set(sample_ids)):
            seen = set()
            duplicate = next(value for value in sample_ids if value in seen or seen.add(value))
            raise ValueError(f"Teacher cache contains duplicate sample_id: {duplicate}")
        if str(raw["split"]).lower() != "train":
            raise ValueError(f"Teacher cache split must be 'train', got {raw['split']!r}")

        logits = torch.as_tensor(raw["logits"]).detach().cpu().float()
        expected_shape = (len(sample_ids), len(EXPECTED_LABEL_TO_ID))
        if tuple(logits.shape) != expected_shape:
            raise ValueError(f"Teacher logits shape must be {expected_shape}, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise ValueError("Teacher logits contain NaN or Inf")

        label_to_id = _normalize_label_mapping(raw["label_to_id"])
        if label_to_id != EXPECTED_LABEL_TO_ID:
            raise ValueError(
                f"Teacher cache label_to_id does not match the fixed five-class mapping: {label_to_id}"
            )
        expected_id_to_label = {index: label for label, index in EXPECTED_LABEL_TO_ID.items()}
        id_to_label = {int(key): str(value) for key, value in raw["id_to_label"].items()}
        if id_to_label != expected_id_to_label:
            raise ValueError(
                f"Teacher cache id_to_label does not match the fixed five-class mapping: {id_to_label}"
            )

        per_sample = {"logits": logits}
        for key in ("feature", "gate"):
            if key not in raw:
                continue
            tensor = torch.as_tensor(raw[key]).detach().cpu().float()
            if tensor.ndim != 2 or tensor.shape[0] != len(sample_ids):
                raise ValueError(
                    f"Teacher {key} must have shape [N, D] with N={len(sample_ids)}, "
                    f"got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Teacher {key} contains NaN or Inf")
            per_sample[key] = tensor

        if "prototypes" in raw:
            prototypes = torch.as_tensor(raw["prototypes"]).detach().cpu().float()
            if prototypes.ndim != 2 or prototypes.shape[0] != len(EXPECTED_LABEL_TO_ID):
                raise ValueError(f"Teacher prototypes have invalid shape: {tuple(prototypes.shape)}")
            if not torch.isfinite(prototypes).all():
                raise ValueError("Teacher prototypes contain NaN or Inf")
            self.global_prototypes = prototypes

        self.by_id = {
            sample_id: {key: tensor[index] for key, tensor in per_sample.items()}
            for index, sample_id in enumerate(sample_ids)
        }
        self.metadata = {
            key: value
            for key, value in raw.items()
            if key not in {"sample_ids", "logits", "feature", "gate", "prototypes"}
        }

    def has_sample(self, sample_id: Any) -> bool:
        return str(sample_id) in self.by_id

    def missing_ids(self, sample_ids: List[Any]) -> List[str]:
        return [str(value) for value in sample_ids if not self.has_sample(value)]

    def get(self, sample_id: Any) -> Dict[str, torch.Tensor]:
        return self.by_id.get(str(sample_id), {})

    def get_required(self, sample_id: Any) -> Dict[str, torch.Tensor]:
        key = str(sample_id)
        if key not in self.by_id or "logits" not in self.by_id[key]:
            raise ValueError(f"Missing teacher logits for sample_id: {key}")
        return self.by_id[key]
