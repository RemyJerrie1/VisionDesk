"""matplotlib canvases embedded in Qt (dark theme). Pure rendering of controller data."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from app.design_system.tokens import COLORS
from core.train import EpochLog

_BG = COLORS["surface"]
_FG = COLORS["text"]
_GRID = COLORS["border"]
_LOSS = COLORS["warning"]
_ACC = COLORS["accent"]
_BEFORE = "#5A657A"  # muted comparison baseline
_AFTER = COLORS["success"]


class Canvas(FigureCanvasQTAgg):
    """A dark-themed matplotlib canvas with helpers to draw the LoRA charts."""

    def __init__(self) -> None:
        self.fig = Figure(figsize=(4, 3), facecolor=_BG, layout="constrained")
        super().__init__(self.fig)

    def _ax(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=8)
        ax.title.set_color(_FG)
        return ax

    def draw_curve(self, history: list[EpochLog], title: str, loss_lbl: str, acc_lbl: str) -> None:
        """Twin-axis training curve: loss (left) + val-acc (right), per epoch."""
        ax = self._ax()
        ax.set_title(title)
        if not history:
            ax.text(0.5, 0.5, "…", ha="center", va="center", color=_FG)
            self.draw_idle()
            return
        epochs = [h.epoch + 1 for h in history]
        ax.plot(epochs, [h.loss for h in history], "-o", color=_LOSS, label=loss_lbl, markersize=4)
        ax.set_xlabel("epoch", color=_FG)
        ax.set_ylabel(loss_lbl, color=_LOSS)
        ax.tick_params(axis="y", colors=_LOSS)
        ax.set_xticks(epochs)

        ax2 = ax.twinx()
        ax2.set_facecolor(_BG)
        accs = [h.val_acc for h in history]
        ax2.plot(epochs, accs, "-s", color=_ACC, label=acc_lbl, markersize=4)
        ax2.set_ylabel(acc_lbl, color=_ACC)
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(axis="y", colors=_ACC, labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color(_GRID)
        self.draw_idle()

    def draw_before_after(
        self, before: float, after: float, title: str, before_lbl: str, after_lbl: str
    ) -> None:
        ax = self._ax()
        ax.set_title(title)
        bars = ax.bar([before_lbl, after_lbl], [before, after], color=[_BEFORE, _AFTER], width=0.5)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("accuracy", color=_FG)
        for b, val in zip(bars, (before, after), strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                val + 0.02,
                f"{val:.2f}",
                ha="center",
                color=_FG,
                fontsize=10,
            )
        self.draw_idle()

    def draw_confusion(
        self, cm: np.ndarray, class_names: list[str], title: str, pred: str, true: str
    ) -> None:
        ax = self._ax()
        ax.set_title(title)
        ax.imshow(cm, cmap="Blues")
        n = len(class_names)
        ax.set_xticks(range(n), [f"{pred}\n{c}" for c in class_names], fontsize=7)
        ax.set_yticks(range(n), [f"{true} {c}" for c in class_names], fontsize=7)
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else _FG,
                    fontsize=10,
                )
        self.draw_idle()
