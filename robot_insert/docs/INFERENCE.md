# Gazebo 推理與自主評估

以下指令在 `/home/skylerkuo/ros2_ws` 執行，預設使用 CUDA、四路相機與已整理的校正資訊。

## 單次執行

```bash
.venv/bin/python -u robot_insert/robot.py run-act --gui true
.venv/bin/python -u robot_insert/robot.py run-flow --gui true
```

| 模式 | 預設 checkpoint |
|---|---|
| ACT | `/mnt/d/robot_models/aic_sfp0_act_direct_scratch_v1/checkpoints/best.pt` |
| ACT＋Flow | `/mnt/d/robot_models/aic_sfp0_flow_recovery_after_direct_act_v1/checkpoints/best.pt` |

可用 `--checkpoint PATH` 指定其他權重。Flow 必須先完成第二階段訓練；ACT 權重不能代替 Flow 配對 checkpoint。
這兩個入口都是純模型自主推理，沒有專家接手。收集模式另見 [收集](COLLECTION.md)。

場景預設每次有小幅隨機變化；指定 seed 方便重現：

```bash
.venv/bin/python -u robot_insert/robot.py run-act --gui true --scene-attempt 42
.venv/bin/python -u robot_insert/robot.py run-flow --gui true --scene-attempt 42 --noise-seed 42
```

`scene-attempt 0` 是官方參考場景；正整數是可重現的小幅 board 位移／旋轉。
Flow 的 `noise-seed` 控制生成噪聲，與場景 seed 不同。
相同 seed 不能保證模擬時序、GPU 計算與接觸物理逐幀完全相同。
`--gui false` 關閉視窗；仍需渲染模型影像。

ACT 預設每個控制時刻重新預測 20 步，對同一執行時刻的重疊預測做時間集成。
Flow 預設 Euler 10 步生成修正，組合 ACT 後採 receding horizon 執行，不套 ACT 的時間集成。
姿態會經四元數正規化、工作範圍及單步位移／旋轉限制；來源控制增益與設定記在每次 run。

## 100 次自主評估

```bash
.venv/bin/python -u robot_insert/robot.py evaluate-act --trials 100 --gui false
.venv/bin/python -u robot_insert/robot.py evaluate-flow --trials 100 --gui false
```

預設結果分別在上述兩個模型目錄的 `evaluation_100_v1`。
每個評估先凍結一份 checkpoint，trial i 使用 `scene-start+i-1`，預設從 1 開始。
評估過程沒有控制器救援；成功必須由官方分數確認。

```bash
# 同一組評估繼續；可切換顯示 GUI
.venv/bin/python -u robot_insert/robot.py evaluate-act --resume --gui true
.venv/bin/python -u robot_insert/robot.py evaluate-flow --resume --gui true
```

未完成的 trial 保持 pending，續跑會先嘗試回收已完成的官方結果。
已完成評估若要另跑新一組，指定新的 `--output-dir /mnt/d/...`，不覆寫舊紀錄。

| 檔案 | 內容 |
|---|---|
| `summary.json` | 完成、成功、失敗、基礎設施錯誤與成功率 |
| `episodes.csv` | 每次場景、官方分數、結果與耗時 |
| `evaluation.json` | 凍結的權重與評估設定 |
| `trials/NNN/launcher.log` | 本次啟動與等待紀錄 |
| `trials/NNN/runs/.../model.log` | 模型、觀測、命令與停止原因 |
| `trials/NNN/runs/.../scoring/scoring.yaml` | 官方結果 |

主要成功率是成功數／所有完成嘗試，包含基礎設施錯誤；另提供只算官方已評分回合的成功率。
等候時間含環境啟動與評分，不完全等於機器手臂的插入時間。

單次 run 也可獨立檢查模型紀錄與評分：

```bash
.venv/bin/python robot_insert/scripts/validate_ftact_run.py /mnt/d/PATH_TO_RUN --require-insertion
```
