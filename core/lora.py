"""Minimal LoRA — hand-rolled (minLoRA isn't on PyPI; peft is HF-heavy).

A LoRALinear keeps a **frozen** base ``Linear`` and adds a **trainable low-rank
adapter** ``B @ A`` (rank r). Only A and B train — that's the whole point of
LoRA: a tiny number of trainable params on top of a frozen model.

    y = base(x) + (x @ Aᵀ @ Bᵀ) * (alpha / r)

Implementing it by hand (not importing a lib) is deliberate — you can explain
exactly what rank / scaling / the frozen base are (exam + interview).
"""

from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        r: int = 8,
        alpha: int = 16,
        base: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.base = base if base is not None else nn.Linear(in_features, out_features)
        for p in self.base.parameters():
            p.requires_grad_(False)  # freeze the base weight
        self.lora_a = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(out_features, r))  # start as no-op
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling


def param_counts(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) — trainable should be a tiny fraction under LoRA."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
