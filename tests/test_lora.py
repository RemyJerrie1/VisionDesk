from __future__ import annotations

import torch

from core.lora import LoRALinear, param_counts


def test_lora_forward_shape() -> None:
    layer = LoRALinear(16, 4, r=2)
    out = layer(torch.randn(5, 16))
    assert out.shape == (5, 4)


def test_base_frozen_adapter_trainable() -> None:
    layer = LoRALinear(16, 4, r=2)
    assert all(not p.requires_grad for p in layer.base.parameters())
    assert layer.lora_a.requires_grad and layer.lora_b.requires_grad


def test_param_counts_trainable_is_small() -> None:
    from core.modelzoo import build_lora_classifier

    model = build_lora_classifier(num_classes=3, r=8)
    trainable, total = param_counts(model)
    assert 0 < trainable < total * 0.02  # LoRA: well under 2% trainable
