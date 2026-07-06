"""Input guards — fail loud with a message the UI can show verbatim."""

from __future__ import annotations


class DataError(ValueError):
    """Raised on invalid/degenerate data the user can fix (too few classes, empty)."""


def check_num_classes(n: int) -> None:
    if n < 2:
        raise DataError(f"need at least 2 classes to train a classifier, got {n}")


def check_non_empty(n_samples: int) -> None:
    if n_samples < 1:
        raise DataError("dataset is empty — no images found")
