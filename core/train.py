"""Train the LoRA adapter (backbone frozen) + evaluate — pure, testable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochLog:
    epoch: int
    loss: float
    val_acc: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / total if total else 0.0


@torch.no_grad()
def confusion(model: nn.Module, loader: DataLoader, num_classes: int) -> np.ndarray:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for x, y in loader:
        pred = model(x).argmax(1)
        for t, p in zip(y.tolist(), pred.tolist(), strict=True):
            cm[t, p] += 1
    return cm


def train_lora(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    on_epoch: Callable[[EpochLog], None] | None = None,
) -> list[EpochLog]:
    """Adam over only the trainable (LoRA) params; returns per-epoch loss/val-acc.

    ``on_epoch`` (optional) fires after each epoch so the UI can update the
    training curve live without the core knowing anything about Qt.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[EpochLog] = []
    for e in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            n += int(y.numel())
        log = EpochLog(epoch=e, loss=running / max(n, 1), val_acc=evaluate(model, val_loader))
        history.append(log)
        if on_epoch is not None:
            on_epoch(log)
    return history
