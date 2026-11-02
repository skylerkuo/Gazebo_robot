# 官方環境與執行邊界

本機使用官方 AIC Gazebo 映像內已建置的 ROS 2 Kilted，與外層 Jazzy workspace 隔離。
透過 `enter.sh` 建立 Linux user／mount／PID namespaces，不需要 Docker。
`COLCON_IGNORE` 防止 colcon 誤把內部 Kilted 套件納入外層建置。

| 路徑 | 用途 |
|---|---|
| `runtime/rootfs/` | 官方映像解壓的可執行環境 |
| `runtime/rootfs.ready` | 環境準備完成標記 |
| `runtime/home/`、`runtime/tmp/` | 隔離執行時的 home／暫存 |
| `runtime/session.lock` | 防止同時啟動兩個 AIC 模擬器工作階段 |
| `tools/` | 既有映像與資料讀取相依套件 |
| `upstream/` | 官方原始碼快照，用來查閱，實際執行 rootfs 版本 |
| `results/` | 歷史官方驗證與診斷證據 |

上述環境與相依檔案不是重複程式，不應作為一般 source 清理目標。
本次 source 整理沒有改動它們，也沒有搬移 WSL、刪除資料集或模型。

## 準備與官方範例

GitHub 原始碼不含 `runtime/`、`tools/`、`upstream/` 或 `results/`。
`prepare-env` 可下載固定映像並還原 rootfs；Python 訓練相依套件需另外安裝。
`environment-tested.json` 記錄本機實際使用的 Python／套件版本，供重建時核對，並非可直接解算的完整 lockfile。
本機 LeRobot 和資料讀取套件是選擇性安裝在 `tools/` 下；LeRobot 0.6.1 的完整套件依賴要求 NumPy 2，
而此 ROS／ML 混合環境使用 NumPy 1.26.4，因此不能把這份版本紀錄直接當成一次 `pip install` 的需求檔。
公開後尚未在乾淨電腦重建並驗證整套環境。

程式使用 LeRobot ACT 模組及官方 AIC 模擬器。若要查閱 upstream 或重新打包資料集資產，
另外取得官方原始碼並切到本文件下方記錄的 commit；正式模擬執行使用固定映像內已建置的版本。

```bash
cd /home/skylerkuo/ros2_ws
.venv/bin/python robot_insert/robot.py prepare-env
bash robot_insert/scripts/run_demo.sh WaveArm true
.venv/bin/python robot_insert/scripts/validate_run.py
bash robot_insert/scripts/run_demo.sh CheatCode true
.venv/bin/python robot_insert/scripts/validate_run.py --require-insertion
```

`prepare_image.py` 優先重用完整 rootfs，不會因過去的 layers 已清除而重抓整個環境。
WaveArm 只驗證手臂運動；CheatCode 使用模擬器真值驗證插接，不是學習模型。
官方範例會使用 sample config 的多個任務；現行資料與訓練僅針對第一個 SFP 插孔。

2026-10-22 的官方 CheatCode 參考驗證為三個固定 trial 全部成功，結果仍保留在
`results/20261022T014036Z_CheatCode_21503/`；這不代表學習模型有相同成功率。

固定映像：

```text
ghcr.io/intrinsic-dev/aic/aic_eval@sha256:253b99a4aa6e0e4fc0cf8c1cfd0269e64aa86411c2bd585c95a1732d4a56a165
```

官方原始碼快照 commit：`6546127c1c15d5786f59c91400f5fc5ad6894448`。
[官方原始碼](https://github.com/intrinsic-dev/aic)；[官方標準安裝說明](https://github.com/intrinsic-dev/aic/blob/main/docs/getting_started.md)。

## 常見等待與空間

首次快取準備會處理影像並做 checksum，之後重用 cache。檢查目前 log 可用：

```bash
.venv/bin/python robot_insert/robot.py status
```

`status` 只反映已寫出的紀錄，不會以舊 log 保證程序還活著。
訓練與收集能並行，但會共用 GPU、RAM 和 D 槽 I/O；只有一個模擬器 session 能持鎖執行。
WSLg 以 D3D12／NVIDIA 渲染，沒有 GUI 也需要相機渲染。

官方 Gazebo／Ruby 結束階段曾產生 WSL crash dump；C 槽曾因此累積大型診斷檔。
磁碟診斷紀錄保留在 `results/`。刪 WSL 內部檔案不保證 Windows 立刻回收 VHD 空間；
壓縮或搬移 WSL 屬於獨立系統維護操作，不能在訓練／收集時執行。
