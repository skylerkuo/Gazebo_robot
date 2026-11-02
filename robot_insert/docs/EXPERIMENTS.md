# 保留的實驗版本與歷史相容

現行推薦入口見 [訓練](TRAINING.md)；以下是可選的比較版本，不是預設工作流。
它們仍由核心 loader／測試支援，所以本次整理沒有刪除模型類別或改變 checkpoint 格式。

所有指令都以 `.venv/bin/python -u robot_insert/scripts/檔名.py` 執行；用 `--help` 查看參數。

| 實驗 | 訓練入口 | 設定 |
|---|---|---|
| 四視角 ACT 混合資料微調 | `train_sfp0_act_multiview.py` | `sfp0_act_expert_finetune.json` |
| PACE Phase＋Event | `train_sfp0_pace_multiview.py` | `sfp0_pace_expert_finetune.json` |
| PACE＋未來 FT 增量 Pred | `train_sfp0_pace_pred_multiview.py` | `sfp0_pace_pred_expert_finetune.json` |
| 全動作 Flow Matching | `train_sfp0_flow_multiview.py` | `sfp0_flow_expert_finetune.json` |
| 舊固定 ACT＋Residual Flow | `train_sfp0_residual_flow_multiview.py` | `sfp0_residual_flow_expert.json` |
| 第二層堆疊 Residual Flow | `train_sfp0_stacked_residual_flow_multiview.py` | `sfp0_stacked_residual_flow_expert.json` |
| ACT／Flow 交替更新 | `train_sfp0_alternating_residual_flow_multiview.py` | `sfp0_alternating_residual_flow_expert.json` |
| 全部資料重新切分、兩者從零 | `train_sfp0_act_flow_fresh.py` | `sfp0_act_flow_fresh_v2.json` |
| 載入完整配對、重新切分、兩者微調 | `train_sfp0_act_flow_finetune.py` | `sfp0_act_flow_finetune_v2.json` |
| 載入舊配對、固定 ACT 只微調 Flow | `train_sfp0_flow_only.py` | `sfp0_flow_only_after_scratch.json` |

上述設定在 `configs/`。部分舊設定依賴之前固定的資料索引／初始化權重；使用前先確認路徑，
不會因保留程式就重新建立已刪除的三視角資料或模型。

## 方法差異

- PACE 版本是在 ACT 上增加 Phase／Event 條件；Pred 額外監督未來 FT 增量。
  這是本專案的適配，不代表完整重現論文；部署仍不提供未來 FT 真值。
- 全動作 Flow 直接生成整段動作；Residual Flow 生成 ACT 的修正量，兩者 checkpoint 不能互換。
- 交替更新每個 cycle 是 ACT BC 一次、重新計算殘差、Flow MSE 一次，總共有兩次 optimizer 更新。
  以相同步數比較單 Flow 與交替更新，不等於相同運算預算。
- 交替訓練可從零或載入配對；載入配對保留完整 Flow，不會只複製 ACT 然後重建生成器。
- 堆疊版固定 ACT＋第一層 Flow，只更新第二層；成本更高，沒有保證更好的結果。

若比較模型，使用同一批固定評估場景、相同 timeout、相同官方成功判準；
若重新分割資料或初始化已看過新 VAL，離線數字不能當作全新未見資料的泛化表現。

## 其他保留能力

`run_sfp0_multiview_gazebo.py --human-control --checkpoint PATH` 可開鍵盤協助收集；
模型／人工來源仍分開記錄。`collect-assisted` 是自動官方專家接手，與人工模式不同。
PACE／Pred／全動作 Flow／堆疊／交替配對仍可透過相應 `evaluate_sfp0_*.py` 評估。
模型載入器保留舊三視角 ACT／PACE／Flow 格式，相關基底與測試仍在。

## 本次原始碼清理

合併了根目錄 28 份重複／歷史 MD，現行說明集中在 README 與本 docs 目錄。
移除無引用的一次性資料檢查、舊三視角推理 CLI、WSL 搬移／壓縮腳本與 GitHub 上傳監控。
對應舊 GitHub 背景程序已停止；目前訓練和資料收集不受此操作影響。

清理前 source、設定與測試已備份到：

```text
/mnt/d/robot_tools/project_archives/robot_insert_source_before_cleanup_20261030T145720.tar.gz
```

備份不包含資料集、模型權重或 runtime。清理報告見 `results/project_cleanup_20261030.json`。
官方快照、rootfs、工具相依、診斷證據與 D 槽模型／資料均保留。
