"""VisionDesk main window — guided 4 steps: observe → prepare data → fine-tune → compare.

Left rail = step navigation (unlocks as prerequisites are met). Right = a per-step
tutorial hint + the step content + Back/Next. States handled: empty · model
loading · training (non-blocking, live curve) · success · error (inline, never crashes).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.charts import Canvas
from app.controller import Observation, TrainResult
from app.i18n import I18n
from app.worker import ObserveWorker, TrainWorker
from core.dataset import Loaders, csv_loaders, folder_loaders, sample_loaders

_FORMAT_FOLDER = "data/\n├─ cat/   a.jpg  b.jpg …\n└─ dog/   c.jpg  d.jpg …"
_FORMAT_CSV = "path,label\nimages/a.jpg,cat\nimages/b.jpg,dog"
_DEMO_LIMIT = 40  # per-class count under which we flag "demo scale"

_QSS = """
QMainWindow, QWidget { background: #0E1117; color: #E6EDF3; }
QFrame#card { background: #1C2128; border: 1px solid #30363D; border-radius: 8px; }
QFrame#hint { background: #16302B; border: 1px solid #2C5A4E; border-radius: 8px; }
QFrame#rail { background: #10141A; border-right: 1px solid #30363D; }
QLabel#hintText { color: #7EE2B8; }
QLabel#h1 { font-size: 16px; font-weight: 600; }
QLabel#big { font-size: 20px; font-weight: 600; }
QLabel#mono { font-family: Consolas, "Courier New", monospace; color: #9DA7B3; }
QLabel#muted { color: #8B949E; }
QLabel#err { color: #E05A52; }
QLabel#step { padding: 8px 10px; border-radius: 6px; color: #8B949E; }
QLabel#stepActive { padding: 8px 10px; border-radius: 6px; background: #1f6feb; color: white; }
QLabel#stepDone { padding: 8px 10px; border-radius: 6px; color: #5DBB7A; }
QPushButton { background: #1f6feb; color: white; border: 0; padding: 8px 14px; border-radius: 6px; }
QPushButton:disabled { background: #30363D; color: #8B949E; }
QSpinBox { background: #1C2128; border: 1px solid #30363D; border-radius: 6px; padding: 4px; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.i18n = I18n("zh")
        self.step = 0  # 0..3
        self.max_step = 0  # highest unlocked step
        self.loaders: Loaders | None = None
        self.observation: Observation | None = None
        self.result: TrainResult | None = None
        self._obs_worker: ObserveWorker | None = None
        self._train_worker: TrainWorker | None = None
        self._live_history: list = []
        self.setStyleSheet(_QSS)
        self.resize(1120, 740)
        self._build()
        self._retranslate()
        self._go(0)

    # ---- layout ----
    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # left rail: step nav + lang + status
        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(210)
        rl = QVBoxLayout(rail)
        self.lang_btn = QPushButton("EN / 中")
        self.lang_btn.clicked.connect(self._toggle_lang)
        rl.addWidget(self.lang_btn)
        rl.addSpacing(8)
        self.step_labels: list[QLabel] = []
        for _ in range(4):
            lbl = QLabel()
            lbl.setObjectName("step")
            self.step_labels.append(lbl)
            rl.addWidget(lbl)
        rl.addStretch(1)
        self.status = QLabel()
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)

        # right: hint + stacked pages + nav
        right = QVBoxLayout()
        hint = QFrame()
        hint.setObjectName("hint")
        hl = QHBoxLayout(hint)
        self.hint_label = QLabel()
        self.hint_label.setObjectName("hintText")
        self.hint_label.setWordWrap(True)
        hl.addWidget(self.hint_label)
        right.addWidget(hint)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_observe())
        self.stack.addWidget(self._page_data())
        self.stack.addWidget(self._page_finetune())
        self.stack.addWidget(self._page_compare())
        right.addWidget(self.stack, 1)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton()
        self.prev_btn.clicked.connect(lambda: self._go(self.step - 1))
        self.next_btn = QPushButton()
        self.next_btn.clicked.connect(lambda: self._go(self.step + 1))
        nav.addWidget(self.prev_btn)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)
        right.addLayout(nav)

        root.addWidget(rail)
        rightw = QWidget()
        rightw.setLayout(right)
        root.addWidget(rightw, 1)

    # ---- page 1: observe ----
    def _page_observe(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.observe_btn = QPushButton()
        self.observe_btn.clicked.connect(self._observe)
        lay.addWidget(self.observe_btn)
        self.obs_head = QLabel()
        self.obs_head.setObjectName("h1")
        lay.addWidget(self.obs_head)
        self.obs_spec = QLabel()
        self.obs_spec.setObjectName("mono")
        self.obs_spec.setWordWrap(True)
        lay.addWidget(self.obs_spec)
        self.obs_top5 = QLabel()
        self.obs_top5.setObjectName("mono")
        self.obs_top5.setWordWrap(True)
        lay.addWidget(self.obs_top5)
        lay.addStretch(1)
        return w

    # ---- page 2: data ----
    def _page_data(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.src_label = QLabel()
        self.src_label.setObjectName("h1")
        lay.addWidget(self.src_label)
        row = QHBoxLayout()
        self.sample_btn = QPushButton()
        self.sample_btn.clicked.connect(self._load_sample)
        self.folder_btn = QPushButton()
        self.folder_btn.clicked.connect(self._load_folder)
        self.csv_btn = QPushButton()
        self.csv_btn.clicked.connect(self._load_csv)
        for b in (self.sample_btn, self.folder_btn, self.csv_btn):
            row.addWidget(b)
        lay.addLayout(row)

        # always-on format-example panel
        fmt = QFrame()
        fmt.setObjectName("card")
        fl = QVBoxLayout(fmt)
        self.fmt_title = QLabel()
        self.fmt_title.setObjectName("h1")
        fl.addWidget(self.fmt_title)
        grid = QHBoxLayout()
        left_col = QVBoxLayout()
        self.fmt_folder_lbl = QLabel()
        self.fmt_folder_lbl.setObjectName("muted")
        fmt_folder = QLabel(_FORMAT_FOLDER)
        fmt_folder.setObjectName("mono")
        left_col.addWidget(self.fmt_folder_lbl)
        left_col.addWidget(fmt_folder)
        right_col = QVBoxLayout()
        self.fmt_csv_lbl = QLabel()
        self.fmt_csv_lbl.setObjectName("muted")
        fmt_csv = QLabel(_FORMAT_CSV)
        fmt_csv.setObjectName("mono")
        right_col.addWidget(self.fmt_csv_lbl)
        right_col.addWidget(fmt_csv)
        grid.addLayout(left_col)
        grid.addLayout(right_col)
        fl.addLayout(grid)
        lay.addWidget(fmt)

        self.preview_label = QLabel()
        self.preview_label.setObjectName("h1")
        lay.addWidget(self.preview_label)
        self.preview = QLabel()
        self.preview.setObjectName("mono")
        self.preview.setWordWrap(True)
        lay.addWidget(self.preview)
        lay.addStretch(1)
        return w

    # ---- page 3: fine-tune ----
    def _page_finetune(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        params = QHBoxLayout()
        self.epochs_lbl = QLabel()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 20)
        self.epochs_spin.setValue(6)
        self.rank_lbl = QLabel()
        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 32)
        self.rank_spin.setValue(8)
        params.addWidget(self.epochs_lbl)
        params.addWidget(self.epochs_spin)
        params.addSpacing(16)
        params.addWidget(self.rank_lbl)
        params.addWidget(self.rank_spin)
        params.addStretch(1)
        lay.addLayout(params)

        self.trainable_lbl = QLabel()
        self.trainable_lbl.setObjectName("big")
        lay.addWidget(self.trainable_lbl)

        self.start_btn = QPushButton()
        self.start_btn.clicked.connect(self._start_training)
        lay.addWidget(self.start_btn)

        self.curve = Canvas()
        lay.addWidget(self.curve, 1)
        return w

    # ---- page 4: compare ----
    def _page_compare(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.acc_lbl = QLabel()
        self.acc_lbl.setObjectName("big")
        lay.addWidget(self.acc_lbl)
        row = QHBoxLayout()
        self.bar = Canvas()
        self.cm = Canvas()
        row.addWidget(self.bar)
        row.addWidget(self.cm)
        lay.addLayout(row, 1)
        return w

    # ---- i18n ----
    def _retranslate(self) -> None:
        t = self.i18n.t
        self.setWindowTitle(t("title"))
        for i, key in enumerate(("step1", "step2", "step3", "step4")):
            self.step_labels[i].setText(t(key))
        self.prev_btn.setText(t("prev"))
        self.next_btn.setText(t("next"))
        self.observe_btn.setText(t("s1_head"))
        self.src_label.setText(t("s2_source"))
        self.sample_btn.setText(t("s2_sample"))
        self.folder_btn.setText(t("s2_folder"))
        self.csv_btn.setText(t("s2_csv"))
        self.fmt_title.setText(t("s2_format"))
        self.fmt_folder_lbl.setText(t("s2_format_folder"))
        self.fmt_csv_lbl.setText(t("s2_format_csv"))
        self.preview_label.setText(t("s2_preview"))
        self.epochs_lbl.setText(t("s3_epochs"))
        self.rank_lbl.setText(t("s3_rank"))
        self.start_btn.setText(t("s3_start"))
        self.acc_lbl.setText(t("s4_accuracy"))

    def _toggle_lang(self) -> None:
        self.i18n.toggle()
        self._retranslate()
        self._render()

    # ---- navigation ----
    def _go(self, step: int) -> None:
        step = max(0, min(3, step))
        if step > self.max_step:
            return
        self.step = step
        self.stack.setCurrentIndex(step)
        self._render()

    def _unlock(self, step: int) -> None:
        self.max_step = max(self.max_step, step)

    def _render(self) -> None:
        t = self.i18n.t
        # step rail styling
        for i, lbl in enumerate(self.step_labels):
            if i == self.step:
                obj = "stepActive"
            elif i < self.max_step:
                obj = "stepDone"
            else:
                obj = "step"
            lbl.setObjectName(obj)
            # force Qt to re-evaluate the objectName-based QSS for this label
            style = lbl.style()
            style.unpolish(lbl)
            style.polish(lbl)

        hint_key = ("s1_hint", "s2_hint", "s3_hint", "s4_hint")[self.step]
        self.hint_label.setText(t(hint_key))
        self.prev_btn.setEnabled(self.step > 0)
        self.next_btn.setEnabled(self.step < self.max_step)

        self._render_observe()
        self._render_data()
        self._render_finetune()
        self._render_compare()

    def _render_observe(self) -> None:
        t = self.i18n.t
        o = self.observation
        if o is None:
            self.obs_head.setText("")
            self.obs_spec.setText("")
            self.obs_top5.setText("")
            return
        self.obs_head.setText(t("s1_head"))
        self.obs_spec.setText(
            f"{t('s1_input')}: {o.spec['input']}\n"
            f"{t('s1_normalize')}: {o.spec['normalize']}\n"
            f"{t('s1_output')}: {o.spec['output']}\n"
            f"{t('s1_params')}: {o.total_params:,}"
        )
        rows = "\n".join(f"{t('s1_class')} #{idx}: {prob:.3f}" for idx, prob in o.top5)
        self.obs_top5.setText(f"{t('s1_top5')}\n{rows}")

    def _render_data(self) -> None:
        t = self.i18n.t
        ld = self.loaders
        if ld is None:
            self.preview.setText("")
            return
        counts = self._class_counts(ld)
        total = sum(counts.values())
        lines = [f"{t('s2_classes')}: {', '.join(ld.class_names)}"]
        lines += [f"  {name}: {counts.get(name, 0)}" for name in ld.class_names]
        if any(c < _DEMO_LIMIT for c in counts.values()):
            lines.append(f"⚠ {t('s2_demo_scale')}")
        lines.append(f"{t('s2_loaded')} ✓  ({total})")
        self.preview.setText("\n".join(lines))

    def _render_finetune(self) -> None:
        t = self.i18n.t
        if self.result is not None:
            r = self.result
            self.trainable_lbl.setText(
                f"{t('s3_trainable')}: {r.trainable:,} {t('s3_of')} {r.total:,} "
                f"({r.trainable_pct:.3f}%)"
            )
            self.curve.draw_curve(r.history, t("s3_curve"), t("s3_loss"), t("s3_acc"))
        else:
            self.trainable_lbl.setText("")

    def _render_compare(self) -> None:
        t = self.i18n.t
        r = self.result
        if r is None:
            self.acc_lbl.setText(t("empty_train"))
            return
        self.acc_lbl.setText(
            f"{t('s4_before')}: {r.before_acc:.2f}   →   {t('s4_after')}: {r.after_acc:.2f}"
        )
        self.bar.draw_before_after(
            r.before_acc, r.after_acc, t("s4_accuracy"), t("s4_before"), t("s4_after")
        )
        self.cm.draw_confusion(
            r.confusion, r.class_names, t("s4_confusion"), t("s4_pred"), t("s4_true")
        )

    # ---- step 1: observe (threaded) ----
    def _observe(self) -> None:
        self.observe_btn.setEnabled(False)
        self._set_status("training", "s1_head")
        self._obs_worker = ObserveWorker()
        self._obs_worker.done.connect(self._on_observed)
        self._obs_worker.failed.connect(self._on_failed)
        self._obs_worker.start()

    def _on_observed(self, obs: Observation) -> None:
        self.observation = obs
        self.observe_btn.setEnabled(True)
        self._unlock(1)
        self._set_status("ready", "ready")
        self._render()

    # ---- step 2: data ----
    def _load_sample(self) -> None:
        self._set_loaders(lambda: sample_loaders(num_classes=3, n_per=24))

    def _load_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self.i18n.t("s2_folder"))
        if path:
            self._set_loaders(lambda: folder_loaders(path))

    def _load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.i18n.t("s2_csv"), "", "CSV (*.csv)")
        if path:
            self._set_loaders(lambda: csv_loaders(path))

    def _set_loaders(self, factory) -> None:
        try:
            self.loaders = factory()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.result = None
        self._unlock(2)
        self._set_status("ready", "s2_loaded")
        self._render()

    # ---- step 3: fine-tune (threaded, live curve) ----
    def _start_training(self) -> None:
        if self.loaders is None:
            self._show_error(self.i18n.t("empty_train"))
            return
        self._set_busy(True)
        self.result = None
        self._live_history = []
        self._set_status("training", "s3_training")
        self._train_worker = TrainWorker(
            self.loaders, epochs=self.epochs_spin.value(), r=self.rank_spin.value()
        )
        self._train_worker.epoch.connect(self._on_epoch)
        self._train_worker.done.connect(self._on_trained)
        self._train_worker.failed.connect(self._on_failed)
        self._train_worker.start()

    def _on_epoch(self, log) -> None:
        self._live_history.append(log)
        t = self.i18n.t
        self.curve.draw_curve(self._live_history, t("s3_curve"), t("s3_loss"), t("s3_acc"))

    def _on_trained(self, result: TrainResult) -> None:
        self.result = result
        self._set_busy(False)
        self._unlock(3)
        self._set_status("ready", "ready")
        self._render()
        self._go(3)

    # ---- shared state ----
    def _set_busy(self, busy: bool) -> None:
        for b in (self.start_btn, self.sample_btn, self.folder_btn, self.csv_btn):
            b.setEnabled(not busy)

    def _set_status(self, kind: str, key: str) -> None:
        self.status.setObjectName("muted")
        self.status.setStyleSheet("")
        prefix = "… " if kind == "training" else "✓ " if key == "ready" else ""
        self.status.setText(prefix + self.i18n.t(key))

    def _show_error(self, message: str) -> None:
        self.status.setObjectName("err")
        self.status.setStyleSheet("color:#E05A52;")
        self.status.setText(self.i18n.t("err_prefix") + message)

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.observe_btn.setEnabled(True)
        self._show_error(message)

    @staticmethod
    def _class_counts(loaders: Loaders) -> dict[str, int]:
        counts = {name: 0 for name in loaders.class_names}
        for _, y in loaders.train:
            for label in y.tolist():
                counts[loaders.class_names[label]] += 1
        for _, y in loaders.val:
            for label in y.tolist():
                counts[loaders.class_names[label]] += 1
        return counts
