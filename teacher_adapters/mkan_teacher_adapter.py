"""Strict integration boundary for a real MKAN-Refine teacher.

The currently available MKAN material is incomplete, so this module deliberately
does not recreate the model.  A real teacher project must provide a small runtime
module with these callables (names are configurable under ``adapter``):

``build_model(config, device) -> torch.nn.Module``
``build_dataloader(dataframe, image_root, batch_size, num_workers, config)``
``forward_logits(model, batch, device, config) -> Tensor[B, 5]``

The runtime module lives in ``teacher_project_dir``.  This adapter loads its real
checkpoint, validates/reorders its class columns, enforces eval mode, and exposes
only logits to the student-side export code.
"""

import importlib
import json
import os
import sys
from collections.abc import Mapping
from typing import Any, Dict

import torch


FIXED_LABEL_TO_ID = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "rescue_volunteering_or_donation_effort": 2,
    "other_relevant_information": 3,
    "not_humanitarian": 4,
}
FIXED_ID_TO_LABEL = {index: label for label, index in FIXED_LABEL_TO_ID.items()}


def load_teacher_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Teacher config not found: {path or 'NOT PROVIDED'}")
    with open(path, "r", encoding="utf-8") as handle:
        if path.lower().endswith(".json"):
            config = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("Install pyyaml to read the teacher config") from exc
            config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("Teacher config must contain a mapping")
    return dict(config)


def read_checkpoint(path: str):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path or "Teacher checkpoint path was not provided")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_top_level_fields(checkpoint) -> list[str]:
    if isinstance(checkpoint, Mapping):
        return sorted(str(key) for key in checkpoint.keys())
    return [f"<{type(checkpoint).__name__}>"]


def _extract_state_dict(checkpoint, state_key: str | None):
    if state_key:
        current = checkpoint
        for part in state_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(f"Configured checkpoint_state_key not found: {state_key}")
            current = current[part]
        state_dict = current
    elif isinstance(checkpoint, Mapping) and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state_dict = checkpoint
    else:
        state_dict = None
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get(key), Mapping):
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            raise ValueError(
                "Could not identify the teacher state dict. Set adapter.checkpoint_state_key in teacher config."
            )
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Teacher state dict is empty or invalid")
    cleaned = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        name = str(key)
        cleaned[name[7:] if name.startswith("module.") else name] = value
    return cleaned


class MKANTeacherAdapter:
    def __init__(
        self,
        teacher_project_dir: str,
        teacher_checkpoint: str,
        teacher_config: str,
        device: torch.device,
    ):
        if not os.path.isdir(teacher_project_dir):
            raise FileNotFoundError(f"Teacher project directory not found: {teacher_project_dir}")
        self.teacher_project_dir = os.path.abspath(teacher_project_dir)
        self.teacher_checkpoint = os.path.abspath(teacher_checkpoint)
        self.teacher_config_path = os.path.abspath(teacher_config)
        self.device = device
        self.config = load_teacher_config(self.teacher_config_path)
        adapter_cfg = self.config.get("adapter", {})
        if not isinstance(adapter_cfg, Mapping):
            raise ValueError("teacher config section 'adapter' must be a mapping")

        runtime_module = adapter_cfg.get("runtime_module")
        if not runtime_module:
            raise ValueError(
                "Teacher config is missing adapter.runtime_module. "
                "A complete MKAN runtime wrapper is required; the model structure is not guessed."
            )
        if self.teacher_project_dir not in sys.path:
            sys.path.insert(0, self.teacher_project_dir)
        self.runtime = importlib.import_module(str(runtime_module))
        self.build_model_fn = self._runtime_callable(adapter_cfg.get("build_model", "build_model"))
        self.build_dataloader_fn = self._runtime_callable(adapter_cfg.get("build_dataloader", "build_dataloader"))
        self.forward_logits_fn = self._runtime_callable(adapter_cfg.get("forward_logits", "forward_logits"))

        checkpoint = read_checkpoint(self.teacher_checkpoint)
        self.checkpoint_fields = checkpoint_top_level_fields(checkpoint)
        teacher_mapping = adapter_cfg.get("teacher_label_to_id")
        if teacher_mapping is None and isinstance(checkpoint, Mapping):
            teacher_mapping = checkpoint.get("label_to_id")
        if not isinstance(teacher_mapping, Mapping):
            raise ValueError(
                "Teacher label mapping is unavailable. Set adapter.teacher_label_to_id in teacher config."
            )
        self.teacher_label_to_id = {str(key): int(value) for key, value in teacher_mapping.items()}
        if set(self.teacher_label_to_id) != set(FIXED_LABEL_TO_ID):
            raise ValueError(f"Teacher classes do not match the fixed five classes: {self.teacher_label_to_id}")
        if sorted(self.teacher_label_to_id.values()) != list(range(5)):
            raise ValueError(f"Teacher class ids must be a permutation of 0..4: {self.teacher_label_to_id}")
        self.column_order = [self.teacher_label_to_id[label] for label in FIXED_LABEL_TO_ID]

        self.model = self.build_model_fn(self.config, self.device)
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("Teacher runtime build_model must return torch.nn.Module")
        state_dict = _extract_state_dict(checkpoint, adapter_cfg.get("checkpoint_state_key"))
        strict = bool(adapter_cfg.get("strict_load", True))
        incompatible = self.model.load_state_dict(state_dict, strict=strict)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "Non-strict teacher load still has incompatible keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.model.to(self.device)
        self.model.eval()
        self.teacher_model_name = str(
            self.config.get("teacher_model_name", adapter_cfg.get("teacher_model_name", type(self.model).__name__))
        )

    def _runtime_callable(self, name: str):
        value = getattr(self.runtime, str(name), None)
        if not callable(value):
            raise AttributeError(f"Teacher runtime callable not found: {name}")
        return value

    def build_dataloader(self, dataframe, image_root: str, batch_size: int, num_workers: int):
        return self.build_dataloader_fn(dataframe, image_root, batch_size, num_workers, self.config)

    def get_teacher_logits(self, batch) -> torch.Tensor:
        self.model.eval()
        output = self.forward_logits_fn(self.model, batch, self.device, self.config)
        if isinstance(output, Mapping):
            output = output.get("logits")
        elif isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("Teacher forward_logits must return a Tensor or an object containing 'logits'")
        if output.ndim != 2 or output.shape[1] != 5:
            raise ValueError(f"Teacher logits must have shape [batch, 5], got {tuple(output.shape)}")
        logits = output.detach()[:, self.column_order]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Teacher logits contain NaN or Inf")
        return logits

    def teacher_forward(self, batch) -> torch.Tensor:
        return self.get_teacher_logits(batch)

    def get_teacher_features(self, batch):
        raise NotImplementedError("Feature KD is intentionally not implemented in this stage")

    def get_teacher_gate(self, batch):
        raise NotImplementedError("Gate KD is intentionally not implemented in this stage")


def load_teacher(
    teacher_project_dir: str,
    teacher_checkpoint: str,
    teacher_config: str,
    device: torch.device,
) -> MKANTeacherAdapter:
    return MKANTeacherAdapter(
        teacher_project_dir=teacher_project_dir,
        teacher_checkpoint=teacher_checkpoint,
        teacher_config=teacher_config,
        device=device,
    )


def teacher_forward(adapter: MKANTeacherAdapter, batch) -> torch.Tensor:
    return adapter.teacher_forward(batch)


def get_teacher_logits(adapter: MKANTeacherAdapter, batch) -> torch.Tensor:
    return adapter.get_teacher_logits(batch)


def get_teacher_features(adapter: MKANTeacherAdapter, batch):
    return adapter.get_teacher_features(batch)


def get_teacher_gate(adapter: MKANTeacherAdapter, batch):
    return adapter.get_teacher_gate(batch)
