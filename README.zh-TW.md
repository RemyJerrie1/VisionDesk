# VisionDesk

**[English](README.md) · 中文**

> 原生桌面應用,用 **LoRA 微調 ResNet50**——凍結預訓練骨幹、掛上輕量低秩
> 轉接層,在筆電 CPU 上就能教它認得**你自己的類別**。引導式四步流程
> (觀察 → 準備資料 → 微調 → 前後對照)讓遷移學習看得見。全程本機、純
> CPU、免費(無 GPU、無付費 API)。

**狀態:** Phase 1 完成 — 核心 + 桌面 UI、已測試、CI 綠、Win/macOS 已打包([Release v0.1.0](https://github.com/RemyJerrie1/VisionDesk/releases/tag/v0.1.0))。

![VisionDesk — 凍結 ResNet50、訓練 LoRA 頭、看 loss/準確率收斂](docs/screenshot.png)

## 為什麼做這個
微調視覺模型常給人「要 GPU、要大資料、要跑別人腳本」的印象。VisionDesk 展示相反的事:**凍結 2350 萬骨幹參數、只訓約 1.6 萬 LoRA 參數(0.070%)**,少量圖片 + CPU 就能提升你自己類別的準確率。每一步都攤在畫面上——模型吃什麼、你的資料要長怎樣、訓練了什麼、LoRA 幫上多少。

## 四步
| 步驟 | 你會看到 |
|---|---|
| ① 觀察原模型 | ResNet50 輸入/輸出規格(3×224×224、ImageNet 正規化 → 1000 類)+ 對範例輸入的 top-5 |
| ② 準備資料 | 內建範例 / 匯入資料夾 / 匯入 CSV——附**常駐格式範例面板** + pandas 預覽(每類張數、demo 規模警告) |
| ③ 掛 LoRA 微調 | epochs / rank 參數、**可訓練 X / 全參數 Y(%)** 標題、即時雙軸訓練曲線(loss + 驗證準確率),訓練在 UI 執行緒外 |
| ④ 前後對照 | 微調前(未訓練頭)vs 後(+LoRA)的準確率 + 混淆矩陣 |

## Phase 1 — 檔案結構
```
core/lora.py      # 自寫 LoRALinear:凍結 base + 可訓練低秩 A/B(y = base(x) + xAᵀBᵀ·α/r)
core/modelzoo.py  # ResNet50(IMAGENET1K_V2)載入/凍結 + LoRA 頭 + top-k 推論 + 輸入規格
core/dataset.py   # 可學習合成範例(免下載)+ ImageFolder + CSV(path,label)
core/train.py     # 只訓轉接層(Adam over 可訓練參數)+ 評估 + 混淆矩陣
core/validate.py  # 資料守門(類別≥2、非空),白話錯誤訊息
app/              # 純 controller + QThread worker + matplotlib 圖 + i18n + 四步視窗
tests/            # LoRA 數學、可訓練<2%、資料守門、真實訓練(合成任務 0.69 → 0.94)
```

## 執行
```bash
python -m venv .venv && ./.venv/Scripts/python -m pip install -r requirements-dev.txt
./.venv/Scripts/python -m app            # 啟動桌面應用
./.venv/Scripts/python -m app --smoke    # 無頭自檢(offscreen、免下載)→ "smoke ok"
./.venv/Scripts/python -m pytest         # 測試
```
> 首次實跑會下載 ResNet50 權重(約 100 MB,一次);`--smoke` 與測試不需要。

## 驗收(Phase 1)
| # | 條件 | 如何驗證 | 狀態 |
|---|------|---------|:--:|
| 1 | LoRA 數學正確 | `pytest`——forward shape、base 凍結 / 轉接層可訓練 | ✅ |
| 2 | 轉接層夠輕 | `pytest`——可訓練 < 全參數 2%(3 類頭 = 0.070%) | ✅ |
| 3 | 真的學得起來 | `pytest`——凍結 ResNet50 + LoRA 頭在合成任務 0.69 → 0.94 | ✅ |
| 4 | 引導式視窗、不卡 UI | `python -m app` → 觀察 → 資料 → 微調(訓練在 UI 外、即時曲線)→ 對照 | ✅ |
| 5 | 資料看得懂 | 常駐資料夾/CSV 格式面板 + pandas 預覽 + demo 規模警告 | ✅ |
| 6 | 可安裝應用 | PyInstaller `.exe`/`.app`,由 `build` workflow(逐 OS `--smoke`) | ✅ Win+macOS 於 CI 打包、逐 OS `--smoke` 通過、已發 Release v0.1.0 |

## 設計取捨
- **自寫 LoRA,不用 `peft`/`minlora`**——轉接層就十來行(`y = base(x) + (x·Aᵀ·Bᵀ)·α/r`、`B` 零初始化使訓練從 base 出發),可解釋、可單元測試,且省下沉重的 Hugging Face 依賴鏈。`minlora` 不在 PyPI、`peft` 以 transformer 為主;單一 `Linear` 頭用自寫版更清楚。反例:要微調 LLM 眾多 attention 層 → 用 `peft`。
- **凍結整個骨幹、Phase 1 只對頭掛 LoRA**——ImageNet 預訓練特徵可遷移,只訓低秩頭正是 CPU + 少量資料可行、且可訓練% 誠實的原因。反例:離 ImageNet 很遠的領域(醫療/衛星)→ 把 LoRA 打進 conv 區塊(後續階段)。
- **可學習的合成範例、以顏色編碼類別**——內建資料把類別編進圖片顏色,讓前後準確率是*真訊號*而非雜訊,且零下載即可跑。反例:要做基準比較 → 匯入真實資料夾/CSV。
- **純核心 + 執行緒化 UI**——所有 ML 邏輯與 Qt 無關且經單元測試;視窗只負責渲染、訓練跑在 `QThread`,所以不會凍結、核心也保持可測。反例:一次性腳本 → 不必拆分。

## 技術棧
Python · **PyTorch / torchvision(ResNet50)** · 自寫 LoRA · numpy/pandas · matplotlib · PySide6 · pytest/ruff/mypy。

> 規格與藍圖:`MyExperience/作品藍圖/作品集/原生桌面/VisionDesk/`(Phase 2 = Grad-CAM 可解釋性,Phase 3 = 物件偵測/YOLO)。
