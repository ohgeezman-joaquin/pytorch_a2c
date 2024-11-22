## 專案介紹

本專案採用 A2C（Advantage Actor-Critic）強化學習演算法，訓練agent在 `LunarLander-v3` 環境中完成著陸任務。本專案提供訓練腳本、模型檢查點儲存功能，以及測試腳本以觀察模型的推理效果。

---

## 專案結構

- **`a2c.py`**：包含 A2C 的核心實現，包括 Actor 和 Critic 網絡架構及更新演算法。
- **`lunar_lander.py`**：封裝了 Lunar Lander 環境的初始化、步驟執行、環境渲染與關閉操作。
- **`main.py`**：訓練腳本，用於訓練 A2C 模型，並儲存檢查點以便後續使用。
- **`test.py`**：測試腳本，負責載入訓練好的模型，並在 Lunar Lander 環境中執行推理與動畫渲染。

---

## 安裝與執行

### 系統需求
- Python 3.8+
- 套件需求：
  - PyTorch
  - Gymnasium
  - Matplotlib
  - Numpy

### 安裝套件
執行以下命令以安裝所需套件：
```bash
pip install torch gymnasium matplotlib numpy
```

### 訓練模型
運行 `main.py` 開始訓練：
```bash
python main.py
```
訓練過程中會每隔一定回合自動儲存檢查點於 `checkpoints` 資料夾中。

### 測試模型
將訓練好的檢查點檔案路徑填入 `test.py` 中的 `checkpoint_path` 變數，運行以下指令觀察推理效果：
```bash
python test.py
```

---

## 檔案功能說明

### `a2c.py`
- 定義 Actor-Critic 網絡架構，使用全連接層與 PReLU 激活函數。
- 實現 A2C 的核心功能，包括策略更新、價值更新與優勢計算。

### `lunar_lander.py`
- 使用 `gymnasium` 建立 Lunar Lander 環境，提供狀態重置、動作執行及環境渲染功能。

### `main.py`
- 負責訓練 A2C 模型，提供總回報的可視化及檢查點儲存功能。
- 目標：agent在環境中取得穩定高回報（如 >= 230 分）。

### `test.py`
- 載入檢查點，並讓訓練好的agent在 Lunar Lander 環境中執行推理。
- 提供動畫渲染，便於直觀觀察agent的動作策略。

---

## 訓練與測試設定

### 訓練參數
- 狀態空間維度：8
- 動作空間維度：4
- 最大步數：300
- 折扣因子：0.99
- Actor 學習率：0.0005
- Critic 學習率：0.0005

### 測試參數
- 檢查點路徑：`checkpoints/a2c_checkpoint_episode_5820.pth`

---

## 視覺化
訓練過程中的總回報將自動繪製為折線圖，存於當前目錄，便於分析模型收斂情況。

---
