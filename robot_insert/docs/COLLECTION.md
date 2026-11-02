# 資料收集與整理

以下指令都在 `/home/skylerkuo/ros2_ws` 執行。資料存到 D 槽。

## 全程控制器直接插入

```bash
.venv/bin/python robot_insert/robot.py status
.venv/bin/python -u robot_insert/robot.py collect --episodes 261 --max-attempts 400 --batch-size 5 --real-time-factor 0.5 --gui false
```

`--episodes` 是原始資料庫的**總合格回合目標**，包含先前的直接插入與修正回合，並不是新增數量。
例如原本總共 211 回合、其中 50 回合為直接插入，目標 261 代表再收 50 個直接插入回合。
請先用 `status` 確認當時數量。重跑相同目標會繼續收集，不需要 `--resume`。

每個 episode 是一次初始接近、對準、下降與官方評分；`batch-size 5` 是同一模擬器程序依序跑 5 個獨立回合。
本模式全程使用官方 CheatCode，不載入 ACT。`--gui true` 顯示 Gazebo；無 GUI 仍需渲染四路相機。
RTF 可改 1.0 試跑；若掉幀或與訓練競爭資源，降回 0.5。

收集位置：`/mnt/d/robot_datasets/aic_sfp0_expert_v1`。
每次命令、影像、state、FT 都保留來源與時序，只有官方評分確認成功及可用記錄才計入合格回合。
成功要求 Tier 1 有效且 Tier 3 明確為 `Cable insertion successful.`；事件通知或部分插入分數不算成功。

未匯入批次會阻止新批次。按錯誤提示恢復對應批次：

```bash
# 全程控制器批次
.venv/bin/python -u robot_insert/robot.py collect --episodes 261 --import-run /mnt/d/robot_datasets/aic_sfp0_expert_v1/runs/RUN_NAME
# ACT＋專家接手批次
.venv/bin/python -u robot_insert/robot.py collect-assisted --import-run /mnt/d/robot_datasets/aic_sfp0_expert_v1/runs/RUN_NAME
```

## ACT 嘗試後由官方專家接手

```bash
.venv/bin/python -u robot_insert/robot.py collect-assisted --episodes 300 --max-attempts 500 --gui false
```

同樣使用總回合目標；預設載入目前從零訓練的直接插入 ACT BEST。可用 `--checkpoint PATH` 指定其他四視角 ACT。
預設 ACT 先嘗試 20 個模擬秒，再由真值幾何控制器接手。接手採當前高度對準 XY／旋轉，再有限下降，不先抬起。
這是時間觸發，不是模型信心判斷。真值只用於專家控制，不會送入模型的觀測。
保留 ACT、expert、hold 的原始來源；接手不會把先前錯誤動作重新標成專家。

## 整理訓練資料

```bash
.venv/bin/python -u robot_insert/robot.py prepare-direct
.venv/bin/python -u robot_insert/robot.py prepare-recovery
```

| 資料夾 | 篩選方式 | 本次已固定的回合數 |
|---|---|---:|
| `aic_sfp0_direct_success_v1` | 全程 CheatCode，官方確認成功，動作全為 expert | 50 |
| `aic_sfp0_recovery_only_v1` | ACT＋專家成功回合，且確實有完整專家修正窗口 | 143 |

兩者都位於 `/mnt/d/robot_datasets/`。來源 episode ID 保留，不要求從 0 重新編號。
Parquet 與已準備的 PNG／數值 cache 優先使用 NTFS 硬連結，校正與評分證明另行整理。
硬連結不是捷徑，刪除原目錄不會刪掉另一目錄的檔案內容；原始資料與 cache 都維持唯讀用途。

整理資料夾的來源清單固定後，不自動混入後續新回合。原始庫增收後，使用**新的整理目錄**：

```bash
.venv/bin/python -u robot_insert/robot.py prepare-direct --output-dir /mnt/d/robot_datasets/aic_sfp0_direct_success_v2
```

再用新資料夾開新訓練，見 [訓練](TRAINING.md)。收集可以與訓練並行，但正在執行的訓練不會即時讀取新回合。

## 時序與過濾

四路 RGB 為左／中／右腕部相機與側相機，20 Hz；記錄尺寸 576×512，訓練 cache 為 RGB 288×256。
每列動作為觀測時間之前最後接受的 TCP 命令；訓練預測 `t+1 ... t+20`，不跨回合、缺幀或控制段。

修正 Flow 只取未來 20 步全為 expert、同段且未被靜止遮罩排除的窗口。
原始軌跡不裁接；長時間無意義停滯由布林遮罩處理，接觸／力變化可能使窗口保留。
目前沒有依插入事件自動截掉完成後命令，相關限制見 [架構與資料語意](ARCHITECTURE.md)。
