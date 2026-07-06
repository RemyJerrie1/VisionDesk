"""Entry point: ``python -m app`` launches the VisionDesk desktop window."""

from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from app.controller import Observation, TrainResult
from app.i18n import I18n
from app.main_window import MainWindow
from core.modelzoo import input_spec
from core.train import EpochLog


def _smoke(window: MainWindow) -> None:
    """Headless self-check: exercise every render path offline (no weight download).

    Feeds a synthetic Observation + TrainResult through the window so the packaged
    .exe/.app is verified to boot and render all four steps without a network.
    """
    window.observation = Observation(spec=input_spec(), total_params=25_557_032, top5=[(281, 0.42)])
    window.result = TrainResult(
        class_names=["cat", "dog"],
        trainable=4_098,
        total=25_557_032,
        history=[EpochLog(0, 1.1, 0.5), EpochLog(1, 0.6, 0.8), EpochLog(2, 0.3, 0.95)],
        before_acc=0.5,
        after_acc=0.95,
        confusion=np.array([[5, 1], [0, 6]]),
    )
    window.max_step = 3
    for step in range(4):
        window._go(step)
    window.i18n = I18n("en")
    window._retranslate()
    window._render()
    print("smoke ok")


def main() -> None:
    smoke = "--smoke" in sys.argv
    if smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication(sys.argv)
    window = MainWindow()

    if smoke:
        _smoke(window)
        return

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
