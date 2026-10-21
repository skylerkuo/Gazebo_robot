# ACT 在 Gazebo 中推理與執行

`act_rollout.py` 載入本專案訓練的 `best.pt`／`last.pt`，輸入即時 RGB 與當前實測關節位置，
輸出未來動作，直接控制模擬手臂與夾爪。模型不需要最終姿態，也不讀取物件座標或 YOLOE 結果。

2026-10-21 起只接受 format 2。請先依 [新版中型訓練](act_training.md) 產生新 checkpoint，
先前 95 分鐘訓練的 format 1 無法繼續載入；資料集與 Gazebo 場景仍可沿用。

## 編譯與啟動場景

先結束舊的 `collect_bottle`／`demo.launch.py`，讓本次 Gazebo 使用相同控制器名稱。
以下編譯使用系統 Python；終端機若已啟用 `.venv`，先執行 `deactivate`。

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select robot_bringup kinova_gen3_lite_support
source install/setup.bash

ros2 launch robot_bringup act_gazebo.launch.py
```

這會開啟原本包含多個物體與箱子的環境，以及與收集資料相同的三個 RGB 視角、初始手臂姿態。
此啟動檔只啟動 Gazebo、robot_state_publisher、bridge 與控制器。
場景開啟後手臂等待推理程式，不會執行 MTC 任務。

瓶子預設 XY=(0.35, 0.15)，可更換為資料收集時附近的位置，例如：

```bash
ros2 launch robot_bringup act_gazebo.launch.py bottle_x:=0.36 bottle_y:=0.14
# 無視窗：附加 headless:=true。
```

瓶子 XY 只用來設定模擬場景，不會輸入 ACT。
每次要重新嘗試，結束推理程式與本次 launch，再重新啟動，恢復手臂、瓶子及其他道具。
在同一個已移動物體的場景重新執行，並不等於新的測試回合。

## 載入模型執行

在第二個終端機啟用原有環境。將 `實際訓練目錄` 換成訓練輸出的時間戳目錄，
明確指定要測試的 checkpoint，避免誤用其他訓練回合：

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src/robot_pick_place/robot_bringup${PYTHONPATH:+:$PYTHONPATH}"

python -m robot_bringup.act_rollout \
  --checkpoint /mnt/d/robot_models/act_bottle_medium/20261021T005811Z_29e0cc88/best.pt \
  --device cuda:0 \
  --precision auto \
  --duration 45
```

這個指令會控制 Gazebo 裡的手臂，長度為 **45 秒模擬時間**，不是 wall time。
若 Gazebo 速度較慢，實際等待會更久；`Ctrl+C` 可提前停止並要求保持當前位置。
`.venv` 提供 PyTorch，ROS setup 提供 rclpy、訊息與 cv_bridge。
建議使用上面的 `python -m`，避免 `ros2 run` 安裝入口所用的系統 Python 缺少 GPU 套件。

只想檢查資料流與推理，可在上述命令附加 `--dry-run --duration 5`。
此模式不送出手臂或夾爪命令；場景中的手臂不會因推理而移動。

不開 Gazebo、只確認模型與前處理，可以先執行：

```bash
python -m robot_bringup.act_rollout \
  --checkpoint /mnt/d/robot_models/act_bottle_medium/20261021T005811Z_29e0cc88/best.pt \
  --device cuda:0 --precision auto --check-model
```

`--check-model` 只載入與暖機，輸出模型設定後結束，不啟動 ROS 或控制器。
也可用 `--device cpu` 檢查模型；CPU 使用 FP32，不需要 ROS setup。
正式推理每次啟動讀取一份完整 checkpoint；若訓練程序之後更新 best.pt，
正在執行的推理不會自動更換權重。重新執行推理才會讀取更新後的模型。

## 新版模型與訓練設定的對應

- 由 format 2 checkpoint 建立同一個 `ACT` 類別，包含微調過的 ResNet18、FrozenBatchNorm、
  DETR encoder／decoder；不再有獨立的舊版推理網路。動作由最後一層 decoder 輸出。
- 從 checkpoint 還原相機順序及影像尺寸。中型模型使用 `scene`、`wrist`、`side` 的 640×480 RGB；
  ROS 影像先轉 RGB，再使用訓練共用的 `image_tensor`，模型內進行相同 ImageNet 正規化。
- qpos 與 action 的平均值、標準差只讀 checkpoint；不從目前 Gazebo 畫面或關節狀態重新估計。
  七維輸出反標準化成 rad 後，才進行 temporal ensembling 與控制命令限制。
- 推理固定 `model.eval()`、`torch.inference_mode()`、`z=0`，不讀未來動作或最終姿態，
  不計算 loss、不更新參數，也不使用訓練用的梯度累積。
- `--precision auto` 在 CUDA 上尊重 checkpoint 的 `no_amp` 設定；允許 AMP 時，
  支援 bf16 就使用 bf16，否則 fp16。CPU 一律 FP32。可明確指定 `--precision fp32` 做數值比較。
  autocast 在推理執行緒內啟用，依 [PyTorch 的執行緒規則](https://docs.pytorch.org/docs/2.11/amp.html#torch.autocast)。
- 開始訂閱 ROS 之前先以人造輸入暖機，所有暖機動作都丟棄。
  啟動輸出與 JSONL configuration 會記錄格式、參數量、影像尺寸、精度、checkpoint 訓練步數、
  標準化來源與 decoder 輸出層。

## 執行流程

1. 由 checkpoint 還原相機順序、影像尺寸、網路大小、關節標準化與時間間隔；驗證 action 標籤版本。
2. 等待 Gazebo robot description、模擬時鐘、啟用中的控制器、三視角影像與關節狀態。
3. 比照收集流程，匹配相機時間差 ≤20ms、因果關節狀態年齡 ≤50ms 的觀測。
4. 推理執行緒每次處理最新可用觀測；處理不及時略過舊觀測，不累積推理工作佇列。
5. 在 10 Hz 模擬時間網格上，對「同一未來時刻」的重疊預測做 temporal ensembling。
   row 0 代表觀測後 0.1 秒；若推理有延遲，選取仍在未來的對應列，不重播過期 row 0。
6. 前六維以單點 `JointTrajectory` 送給手臂控制器，最後一維以 `GripperCommand` 控制夾爪。
   第七維在本機控制器中是 `right_finger_bottom_joint` 的 rad，與訓練一致；不是開關或公尺間距。

時間集成預設衰減 0.01，依原 ACT 對較早的有效預測給較高權重。
可加 `--no-temporal-ensemble` 比較僅使用最新、仍對齊相同未來時間的預測。
設定依據為 [ACT 原始推理程式](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)。
手臂串流使用 [Joint Trajectory Controller 的 topic 介面](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)，
透過控制器狀態與 joint states 監看資料與追蹤誤差。

## 執行限制與紀錄

- 只接受含 `gz_ros2_control/GazeboSimSystem` 的 robot description，這個入口針對 Gazebo Kinova。
- 由 URDF 讀取位置與速度限制。預設額外限制手臂與夾爪位置命令速率為 0.6 rad/s，
  可用 `--max-joint-velocity`／`--max-gripper-velocity` 調整，但不超過 URDF 速度限制。
  這是位置命令的幅度限制，不是硬體速度／加速度保證。
- RGB／控制器資料超過 0.35 模擬秒、關節狀態超過 0.1 模擬秒、時鐘停止、時鐘倒退、
  推理超時、預測用盡、夾爪命令被拒絕或手臂追蹤誤差超過 0.35 rad 時停止。
  停止會要求手臂保持量測位置，並取消目前夾爪動作。
- 預設在 `runs/act_rollout/` 寫入 JSONL：觀測時間、推理耗時、模擬時間延遲、
  預測動作、限制後命令、目前狀態及停止原因。可用 `--log <檔名>` 改存其他位置。
- ACT 本身沒有「抓取完成」輸出，目前以 `--duration` 結束；結束碼 0 只代表推理流程正常完成，
  不代表瓶子成功入箱。需要在 Gazebo 觀察抓取、搬運及放開是否完成。
- 模型直接產生關節命令，目前沒有 MoveIt 碰撞檢查或自動避障，位置／速率限制也不能保證不碰撞。
  是否能成功抓取仍需針對不同瓶子位置進行實際模擬評估。

## 已完成驗證（2026-10-20）

本節是舊版控制通路的歷史驗證；其中 format 1 checkpoint 已不適用新版模型。

- `robot_bringup` 與 `kinova_gen3_lite_support` 編譯成功；專案 78 項單元測試通過。
- 獨立 ROS domain 與 Gazebo partition 中，三視角真實模擬影像、關節狀態、時鐘、控制器與
  ACT 非同步推理可連通；啟動時會等候佇列中的舊影像更新為新資料。
- 載入現有訓練回合 `20261020T075900Z_0c9e68cd/best.pt`，以 CPU 完成 3 秒模擬時間的
  `--dry-run`。模型輸出有限值，沒有送出動作，沒有改動 checkpoint 或重新訓練。
- 使用獨立人造 checkpoint 固定輸出手臂各關節 0.02 rad、夾爪 0.05 rad，完成 3 秒模擬控制，
  由 joint states 確認七個關節到達目標，程式結束時送出保持／取消命令。
  此測試只驗證控制通路，不代表訓練模型的抓取精度或成功率。
- 已關閉驗證用 Gazebo。尚未以訓練模型完成抓瓶入箱或統計成功率。

## 新版中型推理驗證（2026-10-21）

- 專案共 95 項單元測試通過。新增驗證相機順序、RGB／resize、checkpoint 標準化與反標準化、
  z=0、不執行後驗、推理模式無梯度、精度選擇、暖機及 check-model 不啟動 ROS。
- 實際載入中型回合 `20261021T005811Z_29e0cc88/best.pt`：format 2、21,869,319 參數、
  三視角 640×480、20×7 動作、10 Hz。此測試載入的 snapshot 訓練步數為 2398；
  同一檔案後續可能由訓練程序原子更新，推理日誌會記錄每次實際讀取的步數。
- 在 RTX PRO 1000 上，以獨立推理執行緒執行 bf16 forward；實際確認 action head 輸出為 bf16，
  反標準化後回傳有限的 FP32 20×7 動作。未計算 loss 或更新權重。
- 在獨立 Gazebo partition／ROS domain，以 CPU FP32 完成 3 秒模擬時間的實際影像 dry-run，
  收到 6 次預測、25 次動作記錄，沒有送出手臂／夾爪命令。測試用 Gazebo 已關閉。
- 未以此 checkpoint 執行抓瓶入箱，尚未量測成功率。
