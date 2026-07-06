"""ResNet50 loading + freezing + LoRA classifier head + inference.

Phase 1 LoRA scope (documented tradeoff): freeze the whole pretrained backbone,
replace the 1000-class head with a **LoRALinear** (frozen base + trainable
low-rank adapter) sized to the user's classes. The adapter learns the new task
on top of frozen ImageNet features → tiny trainable %. Applying LoRA deeper into
conv blocks is a later phase.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from core.lora import LoRALinear

_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


def input_spec() -> dict[str, str]:
    """What the model eats — shown in the UI so users prep data correctly."""
    return {
        "input": "RGB 3×224×224 tensor",
        "normalize": "ImageNet mean [0.485,0.456,0.406] / std [0.229,0.224,0.225]",
        "output": "1000 ImageNet classes (original) → replaced by your N classes",
    }


def preprocess():
    """torchvision's official ResNet50 transform (resize/crop/normalize)."""
    return _WEIGHTS.transforms()


def load_pretrained() -> nn.Module:
    """Original ResNet50 (1000-class) for the 'observe the model' step."""
    return resnet50(weights=_WEIGHTS)


def build_lora_classifier(num_classes: int, *, r: int = 8, alpha: int = 16) -> nn.Module:
    """Frozen ResNet50 backbone + LoRA head for `num_classes`."""
    model = resnet50(weights=_WEIGHTS)
    for p in model.parameters():
        p.requires_grad_(False)  # freeze backbone
    model.fc = LoRALinear(model.fc.in_features, num_classes, r=r, alpha=alpha)
    return model


@torch.no_grad()
def top_k(model: nn.Module, batch: torch.Tensor, k: int = 5) -> list[list[tuple[int, float]]]:
    """Top-k (class_idx, prob) per image — for the 'observe' step."""
    model.eval()
    probs = torch.softmax(model(batch), dim=1)
    vals, idxs = probs.topk(k, dim=1)
    return [
        [(int(i), float(v)) for i, v in zip(row_i, row_v, strict=True)]
        for row_i, row_v in zip(idxs, vals, strict=True)
    ]
