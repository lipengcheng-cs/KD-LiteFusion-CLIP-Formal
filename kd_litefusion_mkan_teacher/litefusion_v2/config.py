from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class LiteFusionV2Config:
    name: str
    interaction_rank: int = 32
    residual_rank: int = 64
    gate_type: str = "bottleneck"
    gate_hidden: int = 128
    gate_groups: int = 1
    classifier_hidden: int = 128
    dropout: float = 0.2
    feature_dim: int = 768
    num_classes: int = 5
    clip_model_path: str = "/home/lpc/.cache/clip/ViT-L-14-336px.pt"
    freeze_clip: bool = True

    def __post_init__(self) -> None:
        positive = (
            "interaction_rank",
            "residual_rank",
            "gate_hidden",
            "gate_groups",
            "classifier_hidden",
            "feature_dim",
            "num_classes",
        )
        for key in positive:
            if int(getattr(self, key)) <= 0:
                raise ValueError(f"{key} must be positive, got {getattr(self, key)!r}")
        if self.gate_type not in {"legacy", "bottleneck", "grouped"}:
            raise ValueError(f"Unsupported gate_type: {self.gate_type!r}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.gate_type == "grouped":
            if self.feature_dim % self.gate_groups:
                raise ValueError("feature_dim must be divisible by gate_groups")
            if self.gate_hidden % self.gate_groups:
                raise ValueError("gate_hidden must be divisible by gate_groups")
        if not self.freeze_clip:
            raise ValueError("LiteFusion-v2 requires a frozen OpenAI CLIP encoder")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "LiteFusionV2Config":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown LiteFusion-v2 config keys: {unknown}")
        return cls(**dict(values))


_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "v2_a_residual_only": {
        "residual_rank": 64,
        "gate_type": "legacy",
        "gate_hidden": 768,
        "gate_groups": 1,
        "classifier_hidden": 128,
    },
    "v2_p_precision": {
        "residual_rank": 128,
        "gate_type": "bottleneck",
        "gate_hidden": 128,
        "gate_groups": 1,
        "classifier_hidden": 192,
    },
    "v2_b_balanced": {
        "residual_rank": 64,
        "gate_type": "bottleneck",
        "gate_hidden": 128,
        "gate_groups": 1,
        "classifier_hidden": 128,
    },
    "v2_c_compact": {
        "residual_rank": 64,
        "gate_type": "bottleneck",
        "gate_hidden": 64,
        "gate_groups": 1,
        "classifier_hidden": 64,
    },
    "v2_g_grouped": {
        "residual_rank": 64,
        "gate_type": "grouped",
        "gate_hidden": 64,
        "gate_groups": 32,
        "classifier_hidden": 128,
    },
}

CANDIDATE_NAMES = tuple(_CANDIDATES)


def candidate_config(name: str, **overrides: Any) -> LiteFusionV2Config:
    if name not in _CANDIDATES:
        raise KeyError(f"Unknown candidate {name!r}; expected one of {CANDIDATE_NAMES}")
    values: Dict[str, Any] = {
        "name": name,
        "interaction_rank": 32,
        "dropout": 0.2,
        "feature_dim": 768,
        "num_classes": 5,
    }
    values.update(_CANDIDATES[name])
    values.update(overrides)
    return LiteFusionV2Config.from_dict(values)


def load_config(path: str) -> LiteFusionV2Config:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install pyyaml to load LiteFusion-v2 configs") from exc
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    model_values = payload.get("model", payload)
    if not isinstance(model_values, dict):
        raise ValueError(f"model config must be a mapping: {path}")
    return LiteFusionV2Config.from_dict(model_values)
