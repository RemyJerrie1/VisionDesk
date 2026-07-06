"""Datasets: a built-in *learnable* synthetic sample (no download) + ImageFolder + CSV.

The sample encodes class in the image's colour so LoRA can actually learn it →
the before/after accuracy story is real, not noise. Real use = import a folder
(``class_a/*.jpg``) or a CSV (``path,label``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision.datasets import ImageFolder

from core.modelzoo import preprocess
from core.validate import DataError, check_non_empty, check_num_classes


@dataclass(frozen=True)
class Loaders:
    train: DataLoader
    val: DataLoader
    class_names: list[str]


class CsvImageDataset(Dataset):
    """CSV with columns path,label → images decoded + transformed on the fly."""

    def __init__(self, csv_path: str, transform) -> None:
        df = pd.read_csv(csv_path)
        if not {"path", "label"}.issubset(df.columns):
            raise DataError(f"CSV needs columns 'path,label' (has: {list(df.columns)})")
        labels = df["label"].astype(str).tolist()
        self.class_names = sorted(set(labels))
        idx = {c: i for i, c in enumerate(self.class_names)}
        self.paths = df["path"].tolist()
        self.targets = [idx[label] for label in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except OSError as exc:
            raise DataError(f"cannot read image: {self.paths[i]}") from exc
        return self.transform(img), self.targets[i]


def _split(
    dataset: Dataset,
    class_names: list[str],
    *,
    batch: int = 16,
    val_frac: float = 0.25,
    seed: int = 42,
) -> Loaders:
    check_non_empty(len(dataset))  # type: ignore[arg-type]
    check_num_classes(len(class_names))
    n_val = max(1, int(len(dataset) * val_frac))  # type: ignore[arg-type]
    n_train = len(dataset) - n_val  # type: ignore[arg-type]
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    return Loaders(
        train=DataLoader(train_ds, batch_size=batch, shuffle=True),
        val=DataLoader(val_ds, batch_size=batch),
        class_names=class_names,
    )


def sample_loaders(
    num_classes: int = 3, n_per: int = 24, *, size: int = 64, seed: int = 42, **kw
) -> Loaders:
    """Learnable synthetic set: class encoded in colour (64×64, CPU-fast, no files)."""
    gen = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(num_classes):
        for _ in range(n_per):
            img = torch.randn(3, size, size, generator=gen) * 0.3
            img[c % 3] += 1.5  # class-dependent channel shift → learnable signal
            xs.append(img)
            ys.append(c)
    ds = TensorDataset(torch.stack(xs), torch.tensor(ys))
    names = [f"class_{i}" for i in range(num_classes)]
    return _split(ds, names, seed=seed, **kw)


def folder_loaders(root: str, **kw) -> Loaders:
    ds = ImageFolder(root, transform=preprocess())
    return _split(ds, ds.classes, **kw)


def csv_loaders(csv_path: str, **kw) -> Loaders:
    ds = CsvImageDataset(csv_path, preprocess())
    return _split(ds, ds.class_names, **kw)
