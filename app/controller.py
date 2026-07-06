"""Orchestration — pure, Qt-free, unit-testable.

Bridges the guided 4-step UI to core/: ① observe the pretrained ResNet50,
③ attach a LoRA head and fine-tune, ④ report before/after on the user's classes.
Kept UI-free so it can be tested and run off the main thread (see worker.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from core.dataset import Loaders
from core.modelzoo import build_lora_classifier, input_spec, load_pretrained, top_k
from core.train import EpochLog, confusion, evaluate, train_lora


@dataclass(frozen=True)
class Observation:
    """Step ① — what the original ResNet50 eats and produces."""

    spec: dict[str, str]
    total_params: int
    top5: list[tuple[int, float]]  # (imagenet_class_idx, prob) for one sample input


@dataclass(frozen=True)
class TrainResult:
    """Steps ③/④ — the fine-tuning run and the before/after comparison."""

    class_names: list[str]
    trainable: int
    total: int
    history: list[EpochLog]
    before_acc: float  # LoRA head at init (≈ chance on your classes)
    after_acc: float  # after fine-tuning
    confusion: np.ndarray

    @property
    def trainable_pct(self) -> float:
        return 100.0 * self.trainable / self.total if self.total else 0.0


def observe() -> Observation:
    """Load the original 1000-class ResNet50 and run one sample input through it.

    The sample is a deterministic random tensor — its job is to show the *output
    format* (1000-class probabilities → top-5), not to be a meaningful photo.
    """
    model = load_pretrained()
    total = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(0)
    sample = torch.randn(1, 3, 224, 224, generator=gen)
    top5 = top_k(model, sample, k=5)[0]
    return Observation(spec=input_spec(), total_params=total, top5=top5)


def run_training(
    loaders: Loaders,
    *,
    epochs: int = 3,
    r: int = 8,
    lr: float = 1e-2,
    on_epoch: Callable[[EpochLog], None] | None = None,
) -> TrainResult:
    """Freeze ResNet50, attach a LoRA head sized to the data, fine-tune, compare."""
    num_classes = len(loaders.class_names)
    model = build_lora_classifier(num_classes, r=r)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    before = evaluate(model, loaders.val)
    history = train_lora(model, loaders.train, loaders.val, epochs=epochs, lr=lr, on_epoch=on_epoch)
    after = history[-1].val_acc if history else before
    cm = confusion(model, loaders.val, num_classes)

    return TrainResult(
        class_names=loaders.class_names,
        trainable=trainable,
        total=total,
        history=history,
        before_acc=before,
        after_acc=after,
        confusion=cm,
    )
