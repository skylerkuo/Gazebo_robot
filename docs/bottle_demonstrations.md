# 多物體場景：飲料瓶抓放示範資料

使用 Kinova Gen3 Lite、既有 OMPL/RRTConnect 與 MTC Cartesian／夾爪規劃器，
收集「張開夾爪 → 接近 → 夾住 → 抬起 → 搬到箱口 → 放開 → 撤離 → 回 Home」示範。任務文字固定為
`pick up drink bottle`，目標是原場景的綠色 `plastic_bottle`，不是 `drink_can`。
沿用原本 `pick_place.sdf` 的全部物體：橘子、飲料罐、紅色杯子、綠色瓶子、網球與收集箱。
每回合只隨機改變瓶子的起始 XY，其他模型的起始位姿、外觀、碰撞與物理設定都保留原值。
瓶子在箱口上方釋放，再由重力落入箱內，使用既有 `drop_into_bin` 流程。
預設同時收集三路 RGB 影像，用於之後訓練影像式 ACT。

## 執行

在 workspace 根目錄：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select kinova_gen3_lite_support robot_task_core robot_bringup
source install/setup.bash

# 建議先跑 1 回合並開啟 Gazebo，確認物理夾持
ros2 run robot_bringup collect_bottle --episodes 1 --gui

# 收集 10 次嘗試，預設無 GUI，每軸位置偏移 ±2 公分
ros2 run robot_bringup collect_bottle \
  --episodes 10 --xy-jitter 0.02 --seed 42 \
  --output datasets/bottle_pick_place_rgb

# 只測規劃與資料輸出，不執行手臂動作
ros2 run robot_bringup collect_bottle --episodes 1 --plan-only

# 只產生場景、設定與標籤，不啟動 Gazebo／ROS 節點
ros2 run robot_bringup collect_bottle --episodes 2 --prepare-only

# 可選：停用相機，只收集數值狀態
ros2 run robot_bringup collect_bottle --episodes 1 --no-images

# 最終存到 Windows D 槽；動作期間先寫 Linux 暫存，結束後自動轉存
ros2 run robot_bringup collect_bottle --episodes 5 --gui \
  --output /mnt/d/robot_datasets/bottle_pick_place_rgb
```

程式會自行啟動 Gazebo、MoveIt 和記錄器，不必先啟動 demo。
每回合重新啟動完整模擬，還原瓶子、手臂與夾爪狀態，故速度比重用同一場景慢。
它使用獨立的 Gazebo partition 和隨機 ROS domain（100–199），只控制新建的模擬。
如果該 domain 已被其他程式使用，可用 `--domain-id` 指定空的 domain。
每次執行建立新的時間戳／隨機 ID 資料夾，不覆寫舊資料。

## D 槽與本機暫存

收集器會先在 Linux `/tmp/bottle_capture_*/episode_*` 收完一回合，
關閉模擬與 bag 後，才複製到 `--output`。因此 D 槽寫入延遲不會卡住 ROS 訊息接收。
轉存完成後會刪除該回合的 Linux 暫存，再開始下一回合，通常只需暫存一回合的空間。
前次驗證正常回合約 48 MiB，實際大小依動作長度、影像內容與日誌而變化，超時回合可能更大。
可用 `--staging-root /另一個Linux磁碟路徑` 指定暫存位置；不接受 `/mnt` 下的 Windows 磁碟。
`--output` 可繼續指向 `/mnt/d/robot_datasets/bottle_pick_place_rgb`，不需要搬動 ROS 工作區。

轉存先寫隱藏的 `.episode_*.copying-*` 目錄，完成後才改名為正式的 `episode_*`，
並更新 `task.yaml` 的資料路徑。`metadata.json` 的 `storage.state=archived` 表示轉存完成。
複製失敗、D 槽滿或轉存時 Ctrl+C，會保留 Linux 暫存並停止後續收集，終端會顯示可恢復資料的路徑。
動作期間 Ctrl+C 會先結束模擬，再嘗試將當前部分資料轉存，該回合不會列為成功示範。
先確認轉存資料完整，再手動清理意外留下的暫存；不要刪除仍在收集中的目錄。

較早版本直接把 JPEG、每行立即 flush 的索引與 SQLite bag 寫入 `/mnt/d`，
可能因 WSL 跨檔案系統的同步寫入延遲卡住回呼，導致關節狀態、瓶子位姿與影像配對缺口。
已被標為失敗的舊回合維持失敗，不能用放寬同步條件來修補未記錄的訊息。
更新程式後，既有 Python 收集程序仍使用已載入的舊程式，需要結束後重新啟動才能套用修正。

暫存轉存版本已通過 40 項單元測試與一回合完整 Gazebo 抓放驗證：
`/mnt/d/robot_datasets/bottle_storage_validation/20261020T032124Z_3961b62b/episode_0000`。
該回合 `training_candidate=true`、實際入箱檢查通過，執行期間每台 285 張影像與 285 組同步資料，
最大影像／配對間隔 100 ms、三視角時間差 0 ms，共記錄 3,328 筆關節狀態。
D 槽 bag 的 SQLite 完整性檢查通過，轉存完成後 Linux 暫存已清除。
測試也涵蓋轉存中斷或磁碟寫入失敗時保留來源、不發布不完整回合。

預設瓶子中心為 `(0.35, 0.15, 0.06)` 公尺，XY 各均勻偏移 ±0.02 公尺，
高度與朝向固定。可用 `--center-x`、`--center-y`、`--xy-jitter` 調整；
採樣範圍限定在 `x=[0.25,0.45]`、`y=[0.05,0.25]` 公尺內。
這只是場景邊界檢查，並不保證每個位置可規劃／抓取。
採樣的座標會同時寫入 Gazebo 實體瓶子及 MoveIt 物品設定。
原場景的所有模型都保留，原始世界不會被改寫。
橘子、罐子、杯子與網球的實體 collision primitives 會寫入 `known_obstacles.cdr`，
MTC 讀入後與瓶子、支撐面、箱子共同參與碰撞規劃。箱子的五片碰撞幾何會先比對實體世界。
目前已知世界匯入支援這個場景的軸向對齊 box／cylinder／sphere；非零旋轉或 `relative_to` 會拒絕匯入。
`--seed` 控制位置採樣，不保證 OMPL 產生完全相同的軌跡。

`--episodes` 是嘗試數，包含失敗回合。`--timeout` 預設每回合 240 秒（實際時間）；
超時或失敗後保存紀錄，再開始下一回合。Ctrl+C 保存當前部分資料並結束。
全部回合符合要求時退出碼為 0；包含失敗則為 1，使用者中斷則為 130。

## 三路相機與同步

三路皆為 640×480、10 FPS，JPEG 品質 90。場景 RGBD 的原始位置與視角不變。

| 名稱 | 安裝方式 | ROS 原始影像 topic |
|---|---|---|
| `scene` | 原場景相機，固定於 base_link，原位置 `(0.4,-0.6,0.9)` | `/camera_head/color/image_raw` |
| `wrist` | 夾爪底座旁，跟隨夾爪移動；相對 gripper_base_link 位置 `(0.06,0,0.055)` | `/camera_wrist/color/image_raw` |
| `side` | 另一側斜視，固定於 base_link，位置 `(1.0,0.4,0.7)`，涵蓋物體區與箱子 | `/camera_side/color/image_raw` |

腕部與側面相機的方向可在 `kinova_gen3_lite_support/urdf/gen3_lite.urdf.xacro` 調整。
三台 CameraInfo 及 TF 也會保存。這裡的「原視角」指既有 RGBD 感測器的視角，
Gazebo GUI 滑鼠拖動的自由視角不會改變感測器。

開始規劃／動作前會等待三台各至少兩張非空 RGB，避免缺相機時直接執行。
影像訂閱採可靠傳輸，將 JPEG 同時寫入檔案與 bag 的對應 `.../image_raw/compressed` topic；
不再把未壓縮 RGB 重複寫入 bag。原 RGBD 的深度影像／點雲未列入這份訓練資料。

`frames.jsonl` 每行包含三張影像的相對路徑、各自來源時間戳、frame_id、尺寸，
以及固定順序的 7 維 qpos／可用時的 qvel（joint_1…joint_6、right_finger_bottom_joint）。
配對條件是三視角時間差最多 20ms，採用時間不晚於該組最大影像時間戳且相差不超過 50ms 的關節狀態。
若各相機非同時刻拍攝，仍會保留原始時間差；不使用未來關節狀態填補輸入。
缺相機、超出同步容許值或關節狀態太舊的組合不會寫成有效配對。

執行後 `image_check` 會檢查三路影像、三台 calibration 與同步配對是否齊全，
且在完整執行區間內不可有超過 350ms 的影像／配對缺口，通過後才列為訓練候選。
`preview.jpg` 提供同一組三視角預覽（約在執行時間的 1/4，並非固定動作階段）。
`frames.jsonl` 本身是影像／狀態索引，沒有獨立的 action 欄位。
[ACT 訓練程式](act_training.md) 可直接讀取此格式，以嚴格晚於當前影格的實測關節位置建立動作標籤，
不必轉為 HDF5。這是未來實測位置的模仿學習基準；若要學原控制器的命令值，仍需另行整理
控制軌跡、夾爪命令與執行時間的對齊。

## 每回合的檔案

| 檔案 | 內容 |
|---|---|
| `metadata.json` | schema_version=3、任務文字、相機設定、影像完整性、瓶子／箱子設定、執行時間區間與放置判定 |
| `world.sdf`、`task.yaml` | 該回合的實際世界與 MTC 參數，可追溯／檢查 |
| `known_obstacles.cdr` | 由原場景其他物體的碰撞幾何產生的 PlanningSceneWorld，箱子由 MTC 的五片碰撞幾何加入 |
| `solution.cdr`、`solution.json` | 選中的完整 MTC Solution，包含起始場景、手臂與夾爪軌跡、scene diffs |
| `task_description.cdr`、`task_description.json` | stage ID 與名稱，對照 Solution 中的 stage ID |
| `task_status.json` | planned／executing／executed／execution_failed、MoveIt 回傳碼、執行起訖模擬時間（ns） |
| `bag/` | SQLite3 ROS bag：`/joint_states`、手臂 controller reference/feedback、夾爪 action feedback/status、`/clock`、TF、瓶子物理位姿 |
| `bottle_poses.json` | Gazebo 回報的瓶子位置／四元數時間序列，供檢查實際抬升與入箱 |
| `images/scene/`、`images/wrist/`、`images/side/` | 依來源模擬時間（ns）命名的 JPEG |
| `frames.jsonl` | 三視角與當下 7 維關節狀態的時間對齊索引，包含暖機／等待／hold；訓練時依執行區間截取 |
| `calibration/`、`preview.jpg` | 各相機 CameraInfo JSON 與同時間三視角預覽 |
| `launch.log`、`ros_logs/`、`recorder_logs/` | 模擬、MoveIt、記錄器日誌 |

夾爪實際位置包含在 `/joint_states`；action feedback 只在控制器有發布時記錄，
可能為空。靜態 TF 會保留 transient-local QoS，便於 bag 播放。

失敗早於規劃完成時，可能沒有 Solution 或 task status，失敗原因會留在 metadata／log。
`summary.json` 是整次執行的回合索引。
CDR 與 bag 應以相容版本的 ROS 訊息套件讀取；JSON 可直接檢視。
夾爪 action feedback/status 可能出現 rosbag 找不到內嵌 message definition 的警告；
CDR 資料仍可透過已安裝的 ROS 訊息類型反序列化。

Bag 的記錄時間使用收到訊息時最新的 `/clock`，訊息本身保留原始 header timestamp。
執行時間區間使用 MTC 的模擬時鐘。訓練前要依 header timestamp 對齊、重採樣，
截取 `execution_start_ns` 到 `execution_end_ns`，移除啟動／規劃等待與結束 hold 資料，
並固定關節名稱順序；不能直接假設各 topic 的第 N 筆訊息彼此同步。
手臂 controller state 的 `reference` 是當時的控制目標，`feedback` 是實際回饋。
Solution 中夾爪軌跡屬於規劃指令；夾住瓶子時實際夾爪位置可能因接觸／stall 而不同。
若以實際下一時刻關節位置作為 BC 標籤，要明確固定採樣間隔與動作定義。

## 成功標記

`planning_success` 只代表已有選定軌跡；`execution_success` 只代表 MoveIt 回報成功。
`placement_check.verified` 另要求：執行前瓶子位姿接近規劃起點，執行期間
抬高至少 `--minimum-lift`（預設 3 公分），執行後持續至少
`--hold-seconds`（預設 1 秒模擬時間）留在箱內。
入箱判定依瓶子四元數計算圓柱包絡在 XYZ 軸的投影範圍，要求水平包絡在箱子內壁內（容許 3mm 接觸誤差），
底部落在箱底高度 -5mm 至 +20mm 之間；直立／側躺都可，懸在箱口上方、落在箱外或跨過箱壁則不通過。
結束觀察期間位姿回報不能有超過 0.2 秒的缺口。
物理位姿來自 Gazebo PosePublisher，透過 ros_gz_bridge 記錄，與 MoveIt Attach 分開。
此檢查是「實際抬升並在箱內停留」的幾何判定，沒有接觸力感測，不能證明所有情況下都穩定夾持。
可見的拋起、滑落、碰撞仍應人工檢查並排除。

只有執行成功、放入箱內確認通過、必要 topics 有資料、影像完整性通過且未超時的回合，
才標記 `training_candidate: true` 與 `status: placement_verified`。`--plan-only` 永遠不會標成訓練候選。
此版本 `task_type` 為 `pick_place_into_bin`，預設輸出到 `datasets/bottle_pick_place_rgb/`，
與先前只有抬升的 schema_version=1、只有狀態抓放的 schema_version=2 資料分開。
`--no-images` 時不要求影像完整性，也不建立影像檔案。
先用少量回合檢查品質，再大量收集；訓練／驗證／測試以完整回合分組，
保留未見過的位置作測試，避免同一軌跡的相鄰時間點跨集合。

## YOLOE

既有 YOLOE 功能保留。這個收集入口採 `scene_source=known`，不啟動 YOLOE，
用同一份採樣座標控制實體世界與規劃世界，同時保存三路 RGB 與數值狀態。
目前不提供 YOLOE 資料收集旗標，也沒有保存深度。
日後可另收集 YOLOE 估計位姿，並與 Gazebo 真值一併儲存，測試感知誤差下的表現。
把獨立 YOLOE 輸出的物品位姿交給動作策略，不需要採用 VLA。

單一物品／單一文字只訓練這項技能；文字目前作為條件標籤保存，不能據此宣稱策略能理解任意指令。

參考：[Gazebo PosePublisher](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1PosePublisher.html)、
[ros_gz_bridge 訊息對應](https://docs.ros.org/en/jazzy/p/ros_gz_bridge/)。

## 三視角 RGB 驗證（2026-10-20）

- `kinova_gen3_lite_support`、`robot_bringup` 建置通過，全套 36 項單元測試通過。
  新增相機在 URDF→SDF 後保留，腕部相機 parent 為 gripper_base_link。
- 三台曾因 best-effort 大影像傳輸出現掉幀，改用可靠影像訂閱後，完整執行區間三台皆為每 100ms 一張。
- 第一回合完整抓放通過，執行區間每台 288 張、288 組同步資料：
  `/tmp/bottle-three-view-execution/20261020T022326Z_7d62b212/`。
- 最終腕部位置在另一瓶子起點完整驗證：三台各 295 張、295 組同步資料，三視角最大時間差 0ms，
  瓶子實際入箱且保持 1 秒，`training_candidate=true`。
  輸出：`/tmp/bottle-three-view-final/20261020T022505Z_cbfb8f03/`。
- 最終回合所有 JPEG／CompressedImage、CameraInfo、bag 訊息可讀；
  含暖機與 hold 共 1048 張 JPEG、348 組同步索引，整回合資料約 48 MiB（含 JPEG 與 bag 的壓縮影像副本）。
  每組 qpos 為 7 維，狀態時間未晚於影像，這次最大狀態時間差為 0ms。
- 已檢視抓取時三視角：原場景視角不變，腕部可見瓶子與兩側手指，側面可見物件區與箱子。
  最終回合 `preview.jpg` 提供同時間預覽。
- 暖機最早一組缺少當時關節狀態，已丟棄；執行區間無影像缺口。
  本機既有 `move_group` 的清理階段 exit -11 仍存在，任務與資料保存先完成。

尚未進行 ACT 訓練或 GUI 相機面板驗證；目前交付的是三視角資料收集與同步。

## 原場景多物體抓放驗證（2026-10-20）

- `robot_task_core`、`robot_bringup` 建置通過，全套 30 項單元測試通過。
- 檢查產生的世界：除瓶子的起始位姿與位姿發布器外，其他模型與原始 SDF 一致。
- 檢查選定 Solution：起始場景包含瓶子、橘子、飲料罐、紅杯、網球、支撐面與箱子共 7 個碰撞物件；
  stages 包含搬運、放開／Detach、撤離與回 Home。
- 實際 Gazebo 執行兩回合，起點 XY 分別約 `(0.3556, 0.1310)` 與 `(0.3410, 0.1389)` 公尺。
  兩回合均完整執行且通過實際入箱判定：一回合瓶子側躺、一回合直立，均在箱內保持至少 1 秒。
  輸出：`/tmp/bottle-multi-bin-execution/20261020T015723Z_da56bef1/`。
- 第一回合 bag 全部訊息可反序列化，記錄時間順序正確：3931 筆 joint states、3751 筆 controller state、
  1982 筆瓶子位姿、static TF 與夾爪 action status 均有保存。
- 本機既有 `move_group` 在任務完成、資料保存後的 SIGINT 清理階段仍有 exit -11；
  上述執行／入箱結果均在清理前取得，不將清理階段宣稱為完全無錯。

尚未驗證整個採樣區域的成功率、GUI 顯示或模型訓練效果。

## 先前單瓶抬升版本驗證（2026-10-20）

- `robot_task_core`、`robot_bringup` 建置通過；`python3 -m unittest discover -s tests -v` 共 27 項通過。
- `--prepare-only` 兩回合：採樣位置、實體瓶子與規劃位姿一致；重複寫入同一回合會拒絕。
- `--plan-only` 一回合：成功輸出 Solution 與 stage description，確認沒有 place／move home stages。
  輸出：`/tmp/bottle-collection-plan-verified/20261020T014452Z_1cdf52c4/`。
- 實際 Gazebo 執行：seed 42 的兩回合與 seed 7 的一回合均通過執行、抬高約 8 公分、保持 1 秒的判定。
  輸出：`/tmp/bottle-collection-execution-verified/20261020T014553Z_e0b795a5/`、
  `/tmp/bottle-collection-final/20261020T014737Z_5edf9268/`。
- 最終回合 bag 完整讀取、反序列化與時間順序檢查通過：包含 1 筆 static TF、1990 筆 joint states、
  1913 筆 controller state、995 筆瓶子實際位姿，包含夾爪關節與 action status。
- 單元測試確認：只有軌跡成功但瓶子未抬起、抬起後掉落、位姿缺失／缺口、初始位置不符與座標系不符，均不通過抬升判定。

以上是先前單物品抬升版本的模擬驗證，不能用來宣稱新版多物品抓放已成功。
`/tmp` 資料可能被系統清理，正式資料請收集到 workspace 的 `datasets/`。
