"""Render VisionDesk with a synthetic result and save PNGs of each step.

Run on a real display (Windows/macOS): ``python scripts/screenshot.py``.
No weight download / no training — feeds a synthetic TrainResult so the four
steps render deterministically for docs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.controller import Observation, TrainResult  # noqa: E402
from app.i18n import I18n  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from core.modelzoo import input_spec  # noqa: E402
from core.train import EpochLog  # noqa: E402

_OUT = Path(__file__).resolve().parents[1] / "docs"


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.i18n = I18n("en")
    w._retranslate()
    w.observation = Observation(
        spec=input_spec(),
        total_params=25_557_032,
        top5=[(281, 0.31), (285, 0.18), (287, 0.12), (282, 0.09), (283, 0.06)],
    )
    w.result = TrainResult(
        class_names=["cat", "dog", "fox"],
        trainable=16_408,
        total=23_530_587,
        history=[EpochLog(i, 1.1 - i * 0.12, 0.35 + i * 0.09) for i in range(6)],
        before_acc=0.39,
        after_acc=0.92,
        confusion=np.array([[7, 0, 1], [1, 6, 1], [0, 0, 8]]),
    )
    w.max_step = 3
    _OUT.mkdir(exist_ok=True)
    w.show()
    app.processEvents()
    for step in range(4):
        w._go(step)
        app.processEvents()
        w.grab().save(str(_OUT / f"step{step + 1}.png"))
    # main hero shot = the fine-tune step with the training curve
    w._go(2)
    app.processEvents()
    w.grab().save(str(_OUT / "screenshot.png"))
    print(f"saved to {_OUT}")


if __name__ == "__main__":
    main()
