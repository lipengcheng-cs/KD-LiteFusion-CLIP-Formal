import contextlib
import os
from typing import Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import LiteFusionV2Classifier
from .config import LiteFusionV2Config
from .fusion import LiteFusionV2Fusion
from .gate import build_gate


class LiteFusionV2Model(nn.Module):
    """Frozen OpenAI CLIP plus a replaceable LiteFusion-v2 student head."""

    def __init__(
        self,
        config: LiteFusionV2Config,
        device: Union[str, torch.device] = "cpu",
        load_clip: bool = True,
    ):
        super().__init__()
        self.config = config
        self.freeze_clip = bool(config.freeze_clip)
        self.clip: Optional[nn.Module] = None
        self.preprocess = None
        if load_clip:
            self._load_openai_clip(device)

        self.fusion = LiteFusionV2Fusion(
            feature_dim=config.feature_dim,
            interaction_rank=config.interaction_rank,
            residual_rank=config.residual_rank,
            dropout=config.dropout,
        )
        self.gate = build_gate(
            gate_type=config.gate_type,
            feature_dim=config.feature_dim,
            hidden=config.gate_hidden,
            groups=config.gate_groups,
            dropout=config.dropout,
        )
        self.classifier = LiteFusionV2Classifier(
            feature_dim=config.feature_dim,
            hidden=config.classifier_hidden,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    def _load_openai_clip(self, device: Union[str, torch.device]) -> None:
        import clip

        checkpoint = os.path.abspath(os.path.expanduser(self.config.clip_model_path))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"OpenAI CLIP checkpoint not found: {checkpoint}")
        try:
            self.clip, self.preprocess = clip.load(checkpoint, device=device, jit=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load local OpenAI CLIP checkpoint {checkpoint!r}: {exc}") from exc
        actual_dim = int(self.clip.text_projection.shape[1])
        if actual_dim != self.config.feature_dim:
            raise ValueError(
                f"Configured feature_dim={self.config.feature_dim}, but OpenAI CLIP outputs {actual_dim}"
            )
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        self.clip.eval()

    def encode_clip(self, images: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.clip is None:
            raise RuntimeError("encode_clip requires load_clip=True")
        self.clip.eval()
        context = torch.no_grad() if self.freeze_clip else contextlib.nullcontext()
        with context:
            image_feature = F.normalize(self.clip.encode_image(images).float(), dim=-1)
            text_feature = F.normalize(self.clip.encode_text(tokens).float(), dim=-1)
        return image_feature, text_feature

    def forward_head(
        self,
        image_feature: torch.Tensor,
        text_feature: torch.Tensor,
        return_dict: bool = True,
    ):
        image_feature = F.normalize(image_feature.float(), dim=-1)
        text_feature = F.normalize(text_feature.float(), dim=-1)
        fusion_feature = self.fusion(image_feature, text_feature)
        gate = self.gate(image_feature, text_feature, fusion_feature)
        final_feature = fusion_feature + gate * text_feature + (1.0 - gate) * image_feature
        final_feature = F.normalize(final_feature, dim=-1)
        logits = self.classifier(final_feature)
        if not return_dict:
            return logits
        return {
            "logits": logits,
            "feature": final_feature,
            "gate": gate,
            "image_feature": image_feature,
            "text_feature": text_feature,
        }

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        return_dict: bool = True,
    ):
        image_feature, text_feature = self.encode_clip(images, tokens)
        return self.forward_head(image_feature, text_feature, return_dict=return_dict)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.clip is not None and self.freeze_clip:
            self.clip.eval()
        return self

    def head_parameters(self) -> Iterator[nn.Parameter]:
        for module in (self.fusion, self.gate, self.classifier):
            yield from module.parameters()

    def student_state_dict(self) -> Dict[str, torch.Tensor]:
        prefixes = ("fusion.", "gate.", "classifier.")
        return {key: value for key, value in self.state_dict().items() if key.startswith(prefixes)}

    def load_student_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        missing_head = [
            key
            for key in incompatible.missing_keys
            if key.startswith(("fusion.", "gate.", "classifier."))
        ]
        unexpected = list(incompatible.unexpected_keys)
        if missing_head or unexpected:
            raise RuntimeError(f"Invalid LiteFusion-v2 head checkpoint: missing={missing_head}, unexpected={unexpected}")
