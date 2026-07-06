"""End-to-end: freeze ResNet50 + LoRA head actually learns the synthetic task."""

from __future__ import annotations

import torch

from core.dataset import sample_loaders
from core.modelzoo import build_lora_classifier
from core.train import confusion, evaluate, train_lora


def test_lora_head_learns_synthetic() -> None:
    # Seed globally too: build_lora_classifier's frozen base uses the global RNG,
    # so pinning it makes before/after fully reproducible (before=0.69, after=0.94).
    torch.manual_seed(0)
    loaders = sample_loaders(num_classes=2, n_per=32, size=64, batch=8, seed=0)
    model = build_lora_classifier(num_classes=2, r=8)

    before = evaluate(model, loaders.val)
    history = train_lora(model, loaders.train, loaders.val, epochs=8, lr=5e-3)
    after = history[-1].val_acc

    assert len(history) == 8
    # Colour-encoded classes are separable → LoRA should beat chance clearly.
    assert after >= 0.75
    assert after >= before

    cm = confusion(model, loaders.val, num_classes=2)
    assert cm.shape == (2, 2)
