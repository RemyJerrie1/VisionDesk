"""Background fine-tuning worker — keeps the UI responsive during training."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from app.controller import observe, run_training
from core.dataset import Loaders
from core.train import EpochLog


class ObserveWorker(QThread):
    """Loads ResNet50 (first run downloads ~100 MB weights) off the main thread."""

    done = Signal(object)  # Observation
    failed = Signal(str)

    def run(self) -> None:
        try:
            result = observe()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.done.emit(result)


class TrainWorker(QThread):
    """Runs the (CPU-bound) LoRA fine-tuning off the main thread.

    Emits ``epoch(EpochLog)`` after each epoch so the curve updates live, then
    ``done(TrainResult)`` on success or ``failed(str)`` with a user-facing
    message — the window never freezes and never crashes.
    """

    epoch = Signal(object)  # EpochLog
    done = Signal(object)  # TrainResult
    failed = Signal(str)

    def __init__(self, loaders: Loaders, *, epochs: int, r: int) -> None:
        super().__init__()
        self._loaders = loaders
        self._epochs = epochs
        self._r = r

    def run(self) -> None:
        try:
            result = run_training(
                self._loaders,
                epochs=self._epochs,
                r=self._r,
                on_epoch=self._emit_epoch,
            )
        except Exception as exc:  # surface as a message, don't crash the UI
            self.failed.emit(str(exc))
            return
        self.done.emit(result)

    def _emit_epoch(self, log: EpochLog) -> None:
        self.epoch.emit(log)
