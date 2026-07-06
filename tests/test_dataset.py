from __future__ import annotations

import pytest

from core.dataset import sample_loaders
from core.validate import DataError, check_num_classes


def test_sample_loaders_shape() -> None:
    loaders = sample_loaders(num_classes=3, n_per=8, size=32)
    assert loaders.class_names == ["class_0", "class_1", "class_2"]
    x, y = next(iter(loaders.train))
    assert x.ndim == 4 and x.shape[1] == 3  # (batch, 3, H, W)
    assert int(y.max()) < 3


def test_too_few_classes_raises() -> None:
    with pytest.raises(DataError, match="at least 2 classes"):
        check_num_classes(1)
