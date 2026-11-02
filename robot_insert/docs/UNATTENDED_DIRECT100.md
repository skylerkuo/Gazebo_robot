# 控制器 100 回合：無人值守收集與兩階段全新訓練

本次工作目錄：`/mnt/d/robot_tools/aic_direct100_fresh_20261030`。
2026-10-30 執行時，公開版為 211 回合，新增 50 回合只留本機。
2026-11-02 依後續要求補上這 50 回合，公開版已更新為 261 回合，並完成遠端雜湊及匿名下載檢查。

## 執行順序

1. 在 `aic_sfp0_expert_v1` 新增 50 個官方 CheatCode 控制器成功回合，原本 50 個加到 100 個。GUI 關閉、每批 5 回合、RTF 0.5。必須通過官方成功評分及錄製檢查；中斷／不合格紀錄保留但不訓練。
2. 建立 `aic_sfp0_direct_success_100_v1`：只包含這 100 個控制器回合；Parquet 與既有影像快取以硬連結避免重複空間，新回合自動建立快取。
3. ACT＋ResNet-18 完全隨機初始化，訓練 10000 步。95 Train／5 VAL；VAL 固定 1500 個窗口，每 500 步評估與存檔。
4. 凍結本輪 ACT BEST 至工作目錄的 `frozen_act/act.pt`，記錄來源與 SHA-256。
5. 用 `aic_sfp0_recovery_only_v1` 的 143 個合格專家修正回合，訓練新的 Residual Flow 共 20000 步。ACT 固定；Flow 的 ResNet、觀測 Transformer、速度場與其他可訓練權重皆隨機初始化，不讀舊 Flow，也不複製 ACT 編碼器。正規化尺度沿用本輪 ACT，保持組合動作的單位一致。

Flow 設定 `flow_encoder_init_from_act=false`、`pretrained_backbone=false`。這與舊版「從 ACT 複製觀測編碼器」不同；舊設定仍維持相容。
兩階段皆使用 batch 32、CUDA BF16、LR 1e-4／backbone 1e-5、warmup 500、cosine 至 10%。
Flow VAL 為修正資料的 135 Train／8 VAL，固定 1500 個完整專家窗口；不把 model／hold 當成專家答案。

## 新模型目錄

| 模型 | 路徑 |
|---|---|
| 新 ACT | `/mnt/d/robot_models/aic_sfp0_act_direct100_scratch_v1` |
| 新 ACT＋隨機初始化 Flow | `/mnt/d/robot_models/aic_sfp0_flow_recovery_direct100_scratch_v1` |

每個目錄保留 `checkpoints/last.pt`、`best.pt`、`best_position.pt`。舊模型與瓶子專案不變。
`last.pt` 恢復 optimizer、亂數、資料游標與固定 VAL；沒有可續訓 checkpoint 的中斷起跑會先封存，再重新隨機初始化。

## 監測與復原

Windows 的 `supervisor.ps1` 在流程執行期間暫時防止系統睡眠，螢幕仍可關閉；結束後自動釋放，不更改永久電源設定。
Linux 流程每 10 秒更新 heartbeat，每 20 分鐘記錄收集量、訓練步數、VAL、GPU 與 C／D 剩餘空間。
WSL 意外退出時，Windows 監督會嘗試重啟流程並接續資料／checkpoint；完成回合不重收，未完整評分的回合不當成成功。
暫時性檔案讀取錯誤會重試。訓練程序失敗可用 last.pt 重試；重複失敗有次數上限並留下錯誤。

| 檔案 | 內容 |
|---|---|
| `state.json` | 即時階段、PID、heartbeat 與各階段結果 |
| `monitor_latest.json` | 最新監測快照 |
| `monitor.jsonl` | 每 20 分鐘與階段變更的監測歷史 |
| `events.jsonl` | 啟動、重試、復原、完成／錯誤 |
| `collection.log` | 控制器收集及官方評分匯入 |
| `prepare.log` | 100 回合整理與新影像快取 |
| `act.log`、`flow.log` | 模型訓練 |
| `act_config.json`、`flow_config.json` | 本輪實際訓練設定 |
| `windows_supervisor.jsonl` | Windows 監督、WSL 重啟紀錄 |

查看：

```bash
cat /mnt/d/robot_tools/aic_direct100_fresh_20261030/state.json
cat /mnt/d/robot_tools/aic_direct100_fresh_20261030/monitor_latest.json
```

需要停止整個流程時：

```bash
touch /mnt/d/robot_tools/aic_direct100_fresh_20261030/STOP
```

監督會讓目前收集／訓練程序保存退出，並停止後續階段。再次開始前需移除 STOP，確認沒有其他同一流程，再執行 Windows 監督；不要同時啟動多份收集器或訓練。

本次模型推理需明確指定新權重，例如：

```bash
.venv/bin/python -u robot_insert/robot.py run-act --checkpoint /mnt/d/robot_models/aic_sfp0_act_direct100_scratch_v1/checkpoints/best.pt --gui true
.venv/bin/python -u robot_insert/robot.py run-flow --checkpoint /mnt/d/robot_models/aic_sfp0_flow_recovery_direct100_scratch_v1/checkpoints/best.pt --gui true
```

必須等該階段產生 BEST 後才能使用；目前統一入口的預設權重仍指向舊實驗，這裡不自動覆寫。
