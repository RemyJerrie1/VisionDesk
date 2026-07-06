"""Minimal en/zh i18n for the desktop UI (typed dict, keys enforced in parity)."""

from __future__ import annotations

Lang = str  # "en" | "zh"

_DICT: dict[str, dict[str, str]] = {
    "en": {
        "title": "VisionDesk · LoRA fine-tuning of ResNet50",
        # steps
        "step1": "① Observe model",
        "step2": "② Prepare data",
        "step3": "③ Fine-tune (LoRA)",
        "step4": "④ Before / after",
        "prev": "‹ Back",
        "next": "Next ›",
        "ready": "Ready",
        # step 1
        "s1_hint": "First see what the model eats and produces — this dictates how "
        "your data must look.",
        "s1_head": "Original ResNet50 (ImageNet, 1000 classes)",
        "s1_input": "Input",
        "s1_normalize": "Normalize",
        "s1_output": "Output",
        "s1_params": "Total parameters",
        "s1_top5": "Top-5 on a sample input (shows the output format)",
        "s1_class": "class",
        # step 2
        "s2_hint": "Lay your images out like the left panel, then load them in.",
        "s2_source": "Data source",
        "s2_sample": "Built-in sample",
        "s2_folder": "Import image folder…",
        "s2_csv": "Import CSV…",
        "s2_format": "Expected format",
        "s2_format_folder": "Folder layout",
        "s2_format_csv": "CSV layout",
        "s2_preview": "Preview",
        "s2_classes": "Classes",
        "s2_count": "Images per class",
        "s2_demo_scale": "Demo scale — illustrative only.",
        "s2_loaded": "Loaded",
        # step 3
        "s3_hint": "Freeze the backbone, train only LoRA — so a little data on CPU is enough.",
        "s3_epochs": "Epochs",
        "s3_rank": "LoRA rank",
        "s3_trainable": "Trainable params",
        "s3_of": "of",
        "s3_start": "Start fine-tuning",
        "s3_training": "Training… (backbone frozen, LoRA only)",
        "s3_curve": "Training curve",
        "s3_loss": "loss",
        "s3_acc": "val acc",
        # step 4
        "s4_hint": "See how much LoRA improved recognition of your classes.",
        "s4_before": "Before (untrained head)",
        "s4_after": "After (+LoRA)",
        "s4_accuracy": "Accuracy on your classes",
        "s4_confusion": "Confusion matrix (after)",
        "s4_pred": "pred",
        "s4_true": "true",
        # states / errors
        "empty_train": "Prepare data in step ② first.",
        "err_prefix": "⚠ ",
    },
    "zh": {
        "title": "VisionDesk · LoRA 微調 ResNet50",
        "step1": "① 觀察原模型",
        "step2": "② 準備資料",
        "step3": "③ 掛 LoRA 微調",
        "step4": "④ 前後對照",
        "prev": "‹ 上一步",
        "next": "下一步 ›",
        "ready": "就緒",
        "s1_hint": "先看原模型吃什麼、吐什麼——這決定你的資料要長怎樣。",
        "s1_head": "原始 ResNet50(ImageNet,1000 類)",
        "s1_input": "輸入",
        "s1_normalize": "正規化",
        "s1_output": "輸出",
        "s1_params": "總參數量",
        "s1_top5": "對範例輸入的 top-5(展示輸出格式)",
        "s1_class": "類別",
        "s2_hint": "照左邊格式放好,匯入即可。",
        "s2_source": "資料來源",
        "s2_sample": "內建範例",
        "s2_folder": "匯入圖片資料夾…",
        "s2_csv": "匯入 CSV…",
        "s2_format": "格式範例",
        "s2_format_folder": "資料夾式",
        "s2_format_csv": "CSV 式",
        "s2_preview": "預覽",
        "s2_classes": "類別",
        "s2_count": "每類張數",
        "s2_demo_scale": "demo 規模,僅示意。",
        "s2_loaded": "已載入",
        "s3_hint": "凍結骨幹、只訓 LoRA——所以少量資料 + CPU 也能跑。",
        "s3_epochs": "訓練輪數",
        "s3_rank": "LoRA rank",
        "s3_trainable": "可訓練參數",
        "s3_of": "/",
        "s3_start": "開始微調",
        "s3_training": "微調中…(凍結骨幹、只訓 LoRA)",
        "s3_curve": "訓練曲線",
        "s3_loss": "loss",
        "s3_acc": "驗證準確率",
        "s4_hint": "看 LoRA 讓它多認得你的類別多少。",
        "s4_before": "微調前(未訓練頭)",
        "s4_after": "微調後(+LoRA)",
        "s4_accuracy": "在你類別上的準確率",
        "s4_confusion": "混淆矩陣(微調後)",
        "s4_pred": "預測",
        "s4_true": "實際",
        "empty_train": "請先在步驟 ② 準備資料。",
        "err_prefix": "⚠ ",
    },
}

assert _DICT["en"].keys() == _DICT["zh"].keys(), "i18n key parity broken"


class I18n:
    def __init__(self, lang: Lang = "zh") -> None:
        self.lang: Lang = lang if lang in _DICT else "zh"

    def t(self, key: str) -> str:
        return _DICT[self.lang].get(key, key)

    def toggle(self) -> None:
        self.lang = "en" if self.lang == "zh" else "zh"
