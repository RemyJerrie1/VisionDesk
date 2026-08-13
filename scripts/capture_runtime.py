"""Capture the real VisionDesk workflow with a fixed camera.

Uses the cached ResNet50 weights and the actual training pipeline. No mock window
or zoom effect is involved; screenshots are direct QMainWindow grabs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.controller import Observation, run_training  # noqa: E402
from app.i18n import I18n  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from core.dataset import sample_loaders  # noqa: E402
from core.modelzoo import input_spec, load_pretrained, top_k  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "media" / "runtime-frames"


def capture(app: QApplication, window: MainWindow, name: str) -> None:
    app.processEvents()
    window.grab().save(str(OUT / name))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.i18n = I18n("en")
    window._retranslate()
    window._render()
    window.show()

    capture(app, window, "01-start.png")

    model = load_pretrained()
    sample = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    window.observation = Observation(
        spec=input_spec(),
        total_params=sum(p.numel() for p in model.parameters()),
        top5=top_k(model, sample, k=5)[0],
    )
    window._unlock(1)
    window._render()
    capture(app, window, "02-observe.png")

    window.loaders = sample_loaders(num_classes=2, n_per=32, size=64, batch=8, seed=0)
    window._unlock(2)
    window._go(1)
    capture(app, window, "03-data.png")

    window._go(2)
    capture(app, window, "04-configure.png")

    def on_epoch(log) -> None:
        window._live_history.append(log)
        window.curve.draw_curve(window._live_history, "Training curve", "loss", "val acc")
        window.status.setText(f"Training epoch {log.epoch + 1}/8")
        capture(app, window, "05-training.png")

    result = run_training(window.loaders, epochs=8, r=8, lr=5e-3, on_epoch=on_epoch)
    window.result = result
    window._unlock(3)
    window._render()
    capture(app, window, "06-trained.png")
    window._go(3)
    capture(app, window, "07-compare.png")

    evidence = {
        "workflow": "real VisionDesk core + QMainWindow",
        "epochs": len(result.history),
        "classes": result.class_names,
        "trainable_parameters": result.trainable,
        "total_parameters": result.total,
        "trainable_percent": result.trainable_pct,
        "before_accuracy": result.before_acc,
        "after_accuracy": result.after_acc,
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
