# ACT 與 Residual Flow 訓練

## 本輪：100 回合 ACT＋ImageNet，修正 Flow 複製新 ACT 編碼器

這是獨立的新實驗，不載入之前訓練的 ACT／Flow 權重。請按順序執行：

```bash
cd /home/skylerkuo/ros2_ws
.venv/bin/python -u robot_insert/scripts/train_sfp0_act_direct100_imagenet.py
```

ACT 使用 `aic_sfp0_direct_success_100_v1` 的 100 回合控制器直接插入資料，
95 Train／5 VAL。只有 ResNet-18 從 ImageNet 初始化；Transformer 與動作輸出隨機初始化。
ResNet 持續訓練，LR 1e-5；其他 ACT 參數 LR 1e-4。預設 10000 步、warmup 500。
本輪指令不要加 `--from-scratch`，該旗標會關閉 ImageNet。

**ACT 訓練完成後**再執行：

```bash
.venv/bin/python -u robot_insert/scripts/train_sfp0_flow_recovery_transfer.py
```

Flow 僅載入本輪 `aic_sfp0_act_direct100_imagenet_v1/checkpoints/best.pt`：
完整 ACT 固定為 eval／無梯度；Flow 複製其 ResNet、state／FT 投影、視覺投影與觀測 Encoder，
複製後的 Flow 編碼器持續訓練，速度 Decoder／Head 重新初始化，不讀任何舊 Flow。
Flow 使用 `aic_sfp0_recovery_only_v1` 的 143 回合（135 Train／8 VAL），
僅監督完整的專家未來 20 步修正窗口，共 27978 個 Train windows。
原模型動作、hold、跨控制段或不完整窗口不作為 Flow 答案。
Flow 學習專家相對於固定 ACT 的殘差，沿用該 ACT 的正規化；不重新改變 ACT 的尺度。
預設 20000 步、LR 3e-5／backbone 3e-6、warmup 200、Euler ODE 10 步。

兩階段皆 batch 32、CUDA BF16、seed 43、5% 整回合 VAL、固定 1500 anchors，
每 500 步 VAL／保存。位置 loss 的 XYZ 共用 0.01 m 尺度，另輸出三維距離與各軸 MAE／bias／p90（mm）。
ACT 與 Flow 使用不同 VAL 回合，不能直接以兩階段數字宣稱閉迴路成功率提升。

| 階段 | 設定 | 輸出目錄（均在 D 槽） |
|---|---|---|
| ACT | `configs/sfp0_act_direct100_imagenet.json` | `/mnt/d/robot_models/aic_sfp0_act_direct100_imagenet_v1` |
| Flow | `configs/sfp0_flow_recovery_direct100_transfer.json` | `/mnt/d/robot_models/aic_sfp0_flow_recovery_direct100_transfer_v1` |

各指令加 `--check-data` 可唯讀確認資料是否足夠，不會開始訓練；Flow 資料檢查不需要 ACT 已完成。
中斷後在對應指令加 `--resume 該階段輸出目錄/checkpoints/last.pt`，保持資料快照與 optimizer 狀態。
權重、資料快取、ImageNet 快取均在 D 槽；ImageNet 權重已存在 `/mnt/d/robot_cache/torch/hub/checkpoints/`。

以下保留之前的 50 回合從零實驗入口，與本輪輸出分開。

流程：**順利直接插入資料訓練 ACT → 固定 ACT → 專家修正資料訓練 Residual Flow**。
以下在 `/home/skylerkuo/ros2_ws` 執行；先確認其他訓練已保存退出，避免共用 GPU 影響速度。

## 第一階段：ACT 完全從零

```bash
.venv/bin/python -u robot_insert/robot.py train-act --from-scratch --check-data
.venv/bin/python -u robot_insert/robot.py train-act --from-scratch
```

`init_from=null`、`pretrained_backbone=false`：ACT、ResNet-18 均隨機初始化，不讀舊模型或 ImageNet。
state／FT／action 正規化統計僅從本輪 Train 回合計算。`run_config.json` 記錄初始化模式與資料 hash。
目前資料為 `aic_sfp0_direct_success_v1` 的 50 回合，47 Train／3 VAL，seed 43。
每次 VAL 固定取 1500 個不同 anchors，每 500 步評估與保存，預設共 10000 步。

設定：`configs/sfp0_act_direct_success.json`。
輸出：`/mnt/d/robot_models/aic_sfp0_act_direct_scratch_v1`。
預設 batch 32、CUDA／BF16、LR 1e-4／backbone 1e-5、warmup 500、cosine 至 10%。

```bash
# 同一輪中斷後繼續；不要再加 --from-scratch
.venv/bin/python -u robot_insert/robot.py train-act --resume /mnt/d/robot_models/aic_sfp0_act_direct_scratch_v1/checkpoints/last.pt
```

Resume 恢復權重、optimizer、亂數與資料游標，保持同一份資料快照與 VAL。
全新訓練的輸出目錄必須為空；既有結果不會自動覆寫。

## 第二階段：固定 ACT，訓練修正 Flow

**等上述 ACT 訓練完成再執行**，避免拿到尚未定案的 BEST。

```bash
.venv/bin/python -u robot_insert/robot.py train-flow --check-data
.venv/bin/python -u robot_insert/robot.py train-flow
```

啟動時讀取 `aic_sfp0_act_direct_scratch_v1/checkpoints/best.pt`，ACT 完整載入後固定參數與 eval 模式。
Flow 觀測編碼器從該 ACT 複製，殘差生成器重新初始化；整段訓練只更新 Flow。
每批以固定 ACT 重新預測，計算「專家動作－ACT 預測」的修正量。

目前資料為 `aic_sfp0_recovery_only_v1` 的 143 回合：135 Train／8 VAL，27978 個 Train 專家窗口。
VAL 有 1603 個有效窗口，固定取 1500 個。原 ACT 推理、hold 不作為專家答案。
Flow 資料快照可以不同於 ACT，但保留 ACT 正規化尺度與 normalization 來源 hash。

設定：`configs/sfp0_flow_recovery_after_direct_act.json`。
輸出：`/mnt/d/robot_models/aic_sfp0_flow_recovery_after_direct_act_v1`。
預設 10000 步、batch 32、LR 3e-5／backbone 3e-6、warmup 200、每 500 步 VAL／保存。

```bash
.venv/bin/python -u robot_insert/robot.py train-flow --resume /mnt/d/robot_models/aic_sfp0_flow_recovery_after_direct_act_v1/checkpoints/last.pt
```

若已跑滿 10000 步，要延長到累計 20000 步：

```bash
.venv/bin/python -u robot_insert/robot.py train-flow --resume /mnt/d/robot_models/aic_sfp0_flow_recovery_after_direct_act_v1/checkpoints/last.pt --steps 20000
```

延長只改停止步數，保留原本 cosine 排程長度並寫入 checkpoint 的 `lr_schedule_steps`。
到原定 10000 步之後維持最低 LR（主模型 3e-6、backbone 3e-7），不重新 warmup 或突然提高 LR。
資料快照、VAL、固定 ACT、optimizer 與 BEST 紀錄均接續原狀態；不允許用續訓縮短既定總步數。

此 checkpoint 內含 ACT＋Flow 配對權重，部署時不需另行拼接兩個檔案。
若換另一份 ACT 重新開始 Flow，使用新的輸出目錄及 `--init-from ACT_PATH`，重新計算殘差。

## 新資料與新實驗

新收集資料不會自動加入正在跑的訓練。先整理新版資料，再開新輸出：

```bash
.venv/bin/python -u robot_insert/robot.py train-act --from-scratch --dataset-root /mnt/d/robot_datasets/aic_sfp0_direct_success_v2 --output-dir /mnt/d/robot_models/aic_sfp0_act_direct_scratch_v2
```

之後 Flow 必須明確指向該輪 ACT：

```bash
.venv/bin/python -u robot_insert/robot.py train-flow --init-from /mnt/d/robot_models/aic_sfp0_act_direct_scratch_v2/checkpoints/best.pt --output-dir /mnt/d/robot_models/aic_sfp0_flow_recovery_after_direct_act_v2
```

`--from-scratch` 與 `--resume`／`--init-from` 互斥。單純步數從 0 計算不代表隨機初始化，請以 `run_config.json` 的初始化記錄為準。

## 檔案與指標

| 檔案 | 用途 |
|---|---|
| `run_config.json` | 實際設定、初始化、資料 hash、參數與資料數量 |
| `metrics.jsonl` | 每次 Train／VAL／BEST 記錄 |
| `checkpoints/last.pt` | 可續訓權重、optimizer、游標與亂數狀態 |
| `checkpoints/best.pt` | 最低共同尺度 validation loss |
| `checkpoints/best_position.pt` | 最低三維位置誤差 |
| `training_summary.json` | 正常完成或處理停止訊號後的摘要 |

三維位置誤差為每個預測與專家 TCP 目標的歐氏距離，再對樣本與 20 步取平均。
XYZ MAE 是各軸絕對誤差平均；bias 是有方向的平均偏差；p90 是各軸絕對誤差第 90 百分位。
這些是控制目標模仿誤差，不等於插頭到孔的距離或成功率。不同資料切分的 VAL 不能直接排名。
舊交替更新、PACE 等流程見 [實驗版本](EXPERIMENTS.md)。
