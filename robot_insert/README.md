# AIC SFP 插入：控制器收集 → ACT → 修正 Flow

目前主流程是：**控制器收集順利插入 → ACT 完全從零訓練 → 固定 ACT → 用成功修正窗口訓練 Residual Flow → Gazebo 自主評估**。
任務為 `sfp_tip → nic_card_mount_0/sfp_port_0`，四路 RGB、state6＋FT6，預測未來 20 步 TCP 目標。
資料與權重放 D 槽；瓶子專案位於外層 workspace，與本目錄分開。

公開資料：[Kuoskyler/aic-cable-insertion-gazebo-sfp0](https://huggingface.co/datasets/Kuoskyler/aic-cable-insertion-gazebo-sfp0)，
261 回合，包含 100 回合直接插入及 161 回合 ACT 嘗試／專家協助。

目前新實驗使用 **100 回合直接插入 → ImageNet ResNet-18 初始化 ACT → 固定 ACT，複製觀測編碼器給修正 Flow**。
依序執行以下兩個入口；ACT 完成後才啟動 Flow：

```bash
.venv/bin/python -u robot_insert/scripts/train_sfp0_act_direct100_imagenet.py
.venv/bin/python -u robot_insert/scripts/train_sfp0_flow_recovery_transfer.py
```

ACT 10000 步、Flow 20000 步，皆每 500 步評估 1500 個固定 VAL 窗口。
新實驗不讀舊任務權重；ACT 的 ResNet 會微調，Flow 的複製編碼器也會訓練，完整 ACT 在 Flow 階段固定。
設定與續訓方式見 [訓練文件](docs/TRAINING.md)。下方 `robot.py` 常用入口仍對應較早的從零實驗。

GitHub 只提供原始碼與設定。大型執行環境、第三方套件、資料及權重不在這個目錄的 Git 版本內，
需要另外準備；本機驗證版本見 [環境文件](docs/ENVIRONMENT.md) 與 `environment-tested.json`。
設定中的 `/mnt/d`、workspace 路徑及模型來源是本機配置，換電腦時請依 CLI 參數或 JSON 設定修改。

## 統一入口

在 workspace 根目錄執行：

```bash
cd /home/skylerkuo/ros2_ws
.venv/bin/python robot_insert/robot.py --help
.venv/bin/python robot_insert/robot.py status
```

`robot.py` 只路由到既有程式，維持原本 Ctrl+C／存檔行為，不另開背景監控。
原本現行的 `scripts/train_sfp0_act_direct.py`、`train_sfp0_flow_recovery.py`、`collect_sfp0_direct.py` 指令仍可使用。

| 指令 | 用途 |
|---|---|
| `collect` | 全程官方 CheatCode 收集，`--episodes` 是總合格回合目標 |
| `collect-assisted` | ACT 嘗試後由官方專家接手 |
| `prepare-direct` | 整理只有全程控制器成功回合的資料夾 |
| `prepare-recovery` | 整理確實有專家修正窗口的資料夾 |
| `train-act` | 僅直接插入資料，ACT／ResNet 從零訓練 |
| `train-flow` | 固定這輪 ACT BEST，修正資料訓練 Flow |
| `run-act`、`run-flow` | 使用目前主流程的 BEST 自主推理 |
| `evaluate-act`、`evaluate-flow` | 固定權重、100 次場景測試與官方成功率 |
| `status` | 只讀取資料數量及最新訓練 log |

例如 `robot.py train-act --help` 查看該程式參數；
`robot.py --print-command run-flow --gui true` 只預覽實際指令，不啟動模擬器。

## 常用操作

本次已啟動的「新增 50 → 控制器 100 回合全新 ACT → 全新 Flow 20000 步」與每 20 分鐘監測，見 [無人值守流程](docs/UNATTENDED_DIRECT100.md)。

```bash
# 先確認當時回合數；以下 collect 例子目標為原資料庫總共 261 個成功回合
.venv/bin/python -u robot_insert/robot.py collect --episodes 261 --max-attempts 400 --batch-size 5 --gui false

# 已整理好的直接插入資料：ACT、ResNet 完全隨機初始化
.venv/bin/python -u robot_insert/robot.py train-act --from-scratch

# 等 ACT 訓練完成，再固定其 BEST 訓練修正 Flow
.venv/bin/python -u robot_insert/robot.py train-flow

# 推理
.venv/bin/python -u robot_insert/robot.py run-act --gui true
.venv/bin/python -u robot_insert/robot.py run-flow --gui true
```

每 500 步評估固定 1500 個 VAL anchors 並保存，預設 10000 步。
中斷續訓請用 `--resume .../checkpoints/last.pt`；新資料不會自動加入正在訓練的快照。
收集與訓練可並行，但會共用運算與 I/O 資源。

## 資料與模型

| 用途 | D 槽目錄名稱 |
|---|---|
| 收集原始庫 | `robot_datasets/aic_sfp0_expert_v1` |
| 固定直接插入資料（50 回合） | `robot_datasets/aic_sfp0_direct_success_v1` |
| 固定修正資料（143 回合） | `robot_datasets/aic_sfp0_recovery_only_v1` |
| ACT 從零訓練 | `robot_models/aic_sfp0_act_direct_scratch_v1` |
| 固定 ACT＋修正 Flow | `robot_models/aic_sfp0_flow_recovery_after_direct_act_v1` |

這裡的 50／143 是已整理版本的固定清單，不是原始庫的即時數量。
增加資料後，整理到新目錄並開新實驗，詳見收集與訓練文件。
VAL 是動作目標的模仿誤差，不能直接當作插孔對準距離或實際成功率。

## 文件與程式分工

- [環境與官方範例](docs/ENVIRONMENT.md)：rootfs、依賴、WaveArm／CheatCode。
- [收集與整理](docs/COLLECTION.md)：總回合目標、恢復批次、資料篩選。
- [兩階段訓練](docs/TRAINING.md)：從零 ACT、固定 ACT 的 Flow、續訓與新資料。
- [推理與自主評估](docs/INFERENCE.md)：GUI、場景 seed、100 次測試與成功率。
- [架構與資料語意](docs/ARCHITECTURE.md)：輸入、loss、控制目標與限制、核心模組。
- [實驗版本與清理紀錄](docs/EXPERIMENTS.md)：PACE／交替／堆疊等比較及可還原備份。

`scripts/`、`policies/`、`configs/`、`tests/` 為專案原始碼；
`runtime/`、`tools/`、`upstream/` 是執行環境／第三方相依，`results/` 為診斷證據。
名稱較舊的 `ftact.py`、`train_ftact.py` 等仍提供共用功能與權重相容，不能按檔名判斷可刪。

## Hugging Face 資料集發布

公開 repository：[Kuoskyler/aic-cable-insertion-gazebo-sfp0](https://huggingface.co/datasets/Kuoskyler/aic-cable-insertion-gazebo-sfp0)。
新版凍結包位於 `/mnt/d/robot_datasets/_hf_release/aic-cable-insertion-gazebo-sfp0-261-v2`；
包含 261 回合（100 直接插入／161 ACT 加專家）、165044 影格、英文資料卡、
原始來源引用、相機校正及 AIC 資產，不包含訓練權重。
舊 211 回合發布包保留在 `/mnt/d/robot_datasets/_hf_release/aic-cable-insertion-gazebo-sfp0`。

```bash
# 完成後可單獨重驗所有遠端檔案
.venv/bin/python -u robot_insert/scripts/publish_hf_dataset.py \
  --folder /mnt/d/robot_datasets/_hf_release/aic-cable-insertion-gazebo-sfp0-261-v2 \
  --verify-only
```

後續新增資料：先用 `prepare_hf_release.py --previous-release 舊發布包 --output-dir 新發布包`
建立新的凍結版本，再用 `publish_hf_dataset.py --folder 新發布包 --update-from 舊發布包` 上傳。
舊回合的 ID、Parquet、校正及場景檔案必須維持原樣；只追加新檔案。
README 與統計／manifest 會等新檔案傳完後一起提交。中斷時重跑相同指令，只傳缺少或尚未更新的檔案。
若遠端 README 已在網站編輯過，先下載並審閱該版本，使用 `--previous-readme 檔案` 明確指定；
其他未預期的遠端變更仍會阻止更新。

登入憑證、雜湊快取與日誌位於 D 槽 `robot_tools/huggingface`；日誌在其 `publishing` 子目錄。
Xet 的小型工作快取放 `/dev/shm/aic-hf-xet`（記憶體），避開 DrvFs rename/open 錯誤；
每批完成後將新的去重 shard 記錄保存回 D 槽，原始資料始終在 D 槽。
每批完成後校驗遠端 hash，重啟時跳過已完成的回合。
`--upload-batch-size` 預設 8，本輪使用 32，減少小檔案的提交等待。
發布包旁 `aic-cable-insertion-gazebo-sfp0-261-v2-progress.json` 的 `committed_episodes` 是整體已完成回合數。
成功驗證報告為發布包旁的 `aic-cable-insertion-gazebo-sfp0-261-v2-upload-result.json`。
更新中的 README 仍顯示上一版統計，完整資料以新版驗證報告為準。

## 本地驗證

```bash
.venv/bin/python -m pytest -q robot_insert/tests
```

測試以 CPU 小型模型、真實資料格式及假 ROS 驗證程式；不代表已完成真實 Gazebo 成功率測試。
本次整理只移除過時 source 與合併 MD，保留舊權重格式、原資料、環境與現行訓練入口。
