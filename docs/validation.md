# 實際驗證結果

日期：2026-10-19（Asia/Taipei）。目前驗證 myCobot 280 與 Kinova Gen3 Lite。
以下保留各階段的驗證紀錄；最新結果位於最前方小節，歷史未測狀態以最新結果為準。

## 紙盒換成紅色馬克杯（2026-10-19）

移除 small_carton，新增 red_mug，杯身直徑 44mm、高 70mm，位置 `[0.28,-0.25,0.035]`。
紅色杯身、深色杯口與把手均由 SDF primitive 建立；物理碰撞為杯身圓柱，把手僅為視覺。
不包含液體或抓把手功能；其他物品、Profile、AI 閾值、PCL／MTC 均未改動。

- Passed：support 套件增量建置 `/tmp/kinova-mug-build.log`，`gz sdf -k` 回報 Valid。
- Passed：21 項離線測試 `/tmp/kinova-mug-unit.log`，含紙盒移除與馬克杯存在的場景契約。
- Passed：隔離 ROS domain 216 的 headless CPU Gazebo 實際畫面 `/tmp/kinova-mug-camera.png`。
  `red cup` 信心 0.251、`red mug` 0.147，皆選中紅杯；泛稱 `cup`、`mug` 與 `coffee mug` 未檢出。
  記錄 `/tmp/kinova-mug-prompts.log`；這是單張量測，不保證所有視角或擺放。
- Passed：終端輸入 `red mug`，AI confidence=0.147、3D depth_overlap=0.921；
  完整 MTC 規劃與軌跡執行成功，`/tmp/kinova-mug-demo.log`。
- Passed：Gazebo 真實位姿 `[0.25472,-0.51458,0.04500]`，杯身落在收集箱內，
  `/tmp/kinova-mug-poses.txt`。辨識驗證與真實投放均通過；GUI／CUDA 未重測。
- 退出測試時仍出現既有 MoveIt 關閉階段 segmentation fault（exit -11），發生在任務成功與位姿確認之後；此次未修正該既有問題。

## Kinova 更換辨識測試物品（2026-10-16）

移除 apple 與 target_cylinder；保留橘子、飲料罐、紙盒、收集箱。
新增 plastic_bottle（綠色瓶身、瓶頸、藍蓋、淺色標籤，總高 120mm、主體直徑 36mm）與
tennis_ball（黃綠色球體、白色曲線接縫，直徑 52mm）。均由 SDF primitive 建立，無額外資產下載。
known／geometry 預設目標同步改為現存 drink_can，沒有使用已刪除物品的目標座標。

- Passed：兩個整合套件增量建置，`/tmp/kinova-recognizable-props-build.log`。
- Passed：SDF Valid，`/tmp/kinova-recognizable-props-sdf.log`；21 項離線測試，
  `/tmp/kinova-new-props-final-unit.log`，包含已刪除模型不存在、新模型存在、預設目標與實際場景一致。
- Passed：實際 Gazebo RGB 圖片 `/tmp/kinova-new-props-camera.png` 上的 YOLOE 推論。
  `green plastic bottle` 正確選中綠瓶（0.616），`tennis ball` 選中網球（0.387），
  `drink can` 選中藍色罐（0.817）。文字結果 `/tmp/kinova-new-props-recognition.log`、
  `/tmp/kinova-new-bottle-prompts.log`；對應標註影像 `/tmp/kinova-new-props-*.png`。

`bottle`／`water bottle` 泛稱在此畫面會優先選到藍色飲料罐，不能宣稱所有 bottle 提示皆能區分新綠瓶。
此次未改 AI 閾值、PCL／MTC 演算法、抓取角度或 myCobot。

初次綠瓶使用分段碰撞模型時，雖然軌跡返回成功，Gazebo 位姿顯示瓶子彈飛，該次不算物理抓放通過。
已改為與 PCL primitive 一致的保守圓柱包覆碰撞體，慣性與其尺寸一致；控制器與抓取流程未放寬。
初次結果保留 `/tmp/kinova-new-props-initial-demo.log`、`/tmp/kinova-new-props-initial-execute-result.log`。

- Passed（最終碰撞模型）：同一終端依序 `green plastic bottle`、`tennis ball`，
  兩次 3D 對應、完整 MTC 規劃與軌跡執行成功，`/tmp/kinova-new-props-demo.log`、`.cli`、
  `/tmp/kinova-new-props-execute-result.log`。綠瓶 confidence=0.616、depth_overlap=0.998；
  網球 confidence=0.470、depth_overlap=1.000。
- Passed：Gazebo 真實位置確認綠瓶 `[0.25382,-0.51796,0.07000]`、
  網球 `[0.37131,-0.58205,0.03600]`，均落在收集箱內；已移除模型在模擬中不存在。
  記錄 `/tmp/kinova-new-props-final-poses.{txt,json}`。
- Passed：CLI 核心節點唯一性、Controller／服務／Action；測試完成後 quit 清理自己的模擬。
- Passed：更新後預設 drink_can 的 known 模式完整規劃與 CLI 回歸，
  `/tmp/kinova-new-props-known.log`、`.cli`、`-result.log`。

以上為單次 headless CPU 測試；未宣稱所有提示詞／視角皆可可靠辨識，GUI／CUDA 未重測。

## Kinova 開口收集箱投放（2026-10-16）

場景新增外尺寸 32×22×10cm 的開口箱，底面中心 `[0.25,-0.52,0]`。
物品底面在 `[0.25,-0.52,0.18]` 釋放，跳過 lower object，由重力落入箱內。
箱底與四壁均有 Gazebo 碰撞形狀及相同的 MoveIt 五片 BOX；保留原抓取角度與其餘 MTC stage。
myCobot 的 drop_into_bin 預設 false，原放置方式保留。

- Passed：三個整合套件增量建置，`/tmp/kinova-bin-build.log`；21 項離線測試
  `/tmp/kinova-bin-unit.log`；`gz sdf -k` Valid，`/tmp/kinova-bin-sdf.log`。
- Passed：同一終端／同一 MTC 程序依序輸入 `drink can`、`apple`、`ball`，三次完整規劃、
  控制器軌跡執行皆成功。`/tmp/kinova-bin-execute.log`、`.cli`、`-result.log`。
- Passed：Gazebo 真實模型位姿確認飲料罐、蘋果與橘子均在箱內；
  `/tmp/kinova-bin-final-poses.{txt,json}`。最終位置（m）：

  | 物品 | x | y | z |
  |---|---:|---:|---:|
  | 飲料罐 | 0.25090 | -0.51267 | 0.06000 |
  | 蘋果 | 0.37500 | -0.45817 | 0.03500 |
  | 橘色球 | 0.32513 | -0.44601 | 0.03600 |

- Passed：真實 MoveIt 服務中箱體五片 BOX 的尺寸與位姿皆逐一符合 SDF，投放後沒有 attached
  object 或懸空釋放物件。剩下原長圓柱、紙盒、箱體與支撐面四個碰撞物件。
- Passed：CLI 核心節點唯一性、Controller／服務／Action；測試後 quit 清理自己的模擬。
- Passed：原長圓柱的 known 模式也能完成整個投放規劃，物品中心高度自動換算為 0.355m；
  `/tmp/kinova-bin-known-plan.log`、`.cli`、`-result.log`。此項 execute=false，未宣稱圓柱物理投放成功。
- Passed：myCobot 原 known 模式完整規劃及 CLI 回歸，
  `/tmp/mycobot-bin-regression.log`、`.cli`、`-result.log`。

目前箱子在抓取 crop 之外，箱內已收集物件不重建為待抓目標。
尚未宣稱紙盒或原長圓柱的物理投放成功；箱子容量有限，未實作裝滿偵測或堆疊規劃。
原 MoveIt 2.12.4 關閉崩潰仍存在，本次成功任務不代表該既有問題已修正；GUI／CUDA 未重測。

最初箱體 visual／collision 同名的 SDF 檢查錯誤已更正，最終 SDF Valid。
第一次檢查器誤將剩餘碰撞物件數量預期為 3，改成正確的 4 後，真實服務幾何檢查通過；
原檢查輸出保留於 `/tmp/kinova-bin-execute-initial-check.log`。

## AI 終端輸入、請求後取像與重複任務（2026-10-16）

- Passed：`robot_task_interfaces`、`robot_perception`、`robot_task_core`、`robot_bringup`
  建置，`/tmp/robot-interactive-build.log`；後續核心修正建置另見
  `/tmp/robot-interactive-core-fix-build.log`、`/tmp/robot-interactive-client-reuse-build.log`。
- Passed：20 項 Profile／模型／AI 對應檢查，`/tmp/robot-interactive-final-unit.log`；
  9 項真實 MTC 無效設定／場景逾時／場景失敗檢查，`/tmp/robot-interactive-safety.log`。
- Passed：真實 Kinova 模擬與 YOLOE 下，等待／空白輸入皆不觸發取像、YOLOE、PCL 或 MTC。
  忙碌時第二筆請求被立即拒絕。`drink can` 完整規劃後，`purple helicopter` 辨識失敗，
  接著 `drink can` 再完成一次完整規劃。三次取像的 RGB 與點雲時間戳均晚於各自請求。
  `/tmp/kinova-ai-commands-final.log`、`.cli`、`-result.log`。最終版本重測已通過，
  重用同一個場景 client，沒有重複 node name 警告。
- Passed：感知服務沒有新畫面時會逾時；取得一組新畫面後，下一次沒有新畫面仍須逾時，
  不重用前一組資料。真實 ROS 感知服務／模擬相機發布器檢查，`/tmp/robot-fresh-capture-check.log`。
- Passed：透過真正 `pick_console` 終端 stdin，依序輸入 `drink can` → `apple` → `drink can`。
  兩次飲料罐皆完整規劃及 Gazebo 執行成功，MTC 程序與模型只啟動一次；apple 規劃失敗後仍可接續。
  `/tmp/kinova-ai-console-execute.log`、`.cli`、`-result.log`。測試入口：

  ```bash
  ros2 run robot_bringup pick_console --robot kinova_gen3_lite --execute \
    --headless --no-rviz --log /tmp/kinova-ai-console-execute.log
  ```

- Passed：Gazebo `/world/default/pose/info` 顯示第一次執行後，真實罐子由
  `[0.30,-0.02,0.05]` 移到 `[0.34646,-0.14383,0.05]`，接近原放置點 `[0.35,-0.15]`；
  原長圓柱與其他物品位姿未變。記錄 `/tmp/kinova-ai-console-after-first-poses.txt`。
- Passed：終端測試完成後輸入 `quit`，由入口清理自己啟動的模擬。
- Passed：最終程式的原 Kinova 幾何飲料罐規劃、myCobot known 規劃及 CLI 檢查，
  `/tmp/kinova-interactive-geometry-regression.log`、`/tmp/mycobot-interactive-regression.log`，
  各有 `.cli` 與 `-result.log`。
- Passed：`interactive=false` 的原單次 AI 完整規劃與 CLI 檢查，
  `/tmp/kinova-interactive-one-shot-regression.log`、`.cli`、`-result.log`。

apple 的 AI 辨識與 3D 對應成功（confidence=0.110、depth_overlap=1.000），
但 `place pose IK (0/460)` 回報 `cylinder_0 - left_finger_dist_link` 碰撞：
前一個罐子已占據固定放置點。此項不算 apple 抓放成功，也未自動更改放置點或放寬碰撞。
保留使用者的 20° 抓取設定與原 MTC stage，未新增多物品自動排列功能。

既有 MoveIt 2.12.4 關閉／執行期崩潰並未在此功能中修正；本次兩次飲料罐執行皆先成功，
不能推論其他場景的既有崩潰已消失。GUI／CUDA 本次未重測。

重現等待／busy／失敗重試／重複規劃測試：

```bash
python tests/check_ai_commands.py
```

## YOLOE／幾何匹配切換（2026-10-16）

- Passed：`robot_perception`、`robot_bringup` 增量建置；`/tmp/robot-ai-integration-build.log`。
- Passed：20 項 Profile／模型／AI 深度對應測試；`/tmp/robot-ai-final-unit-tests.log`。
  包含 organized cloud 的像素位置、row padding、endianness、NaN、時間差、TF、
  排除支撐面、唯一目標與歧義拒絕。
- Passed：使用真實 Ultralytics Results 驗證每類最高信心選擇保留相同實例的 mask，空結果可處理。
- Passed：Kinova 原幾何模式，飲料罐完整 Pick and Place 規劃與 CLI 檢查；
  `/tmp/kinova-geometry-ai-regression.log`、`.cli`、`-result.log`。
- Passed：myCobot 原 known 模式完整規劃與 CLI 檢查；
  `/tmp/mycobot-ai-regression.log`、`.cli`、`-result.log`。
- Passed：真實 Kinova Gazebo 相機 → YOLOE → 遮罩深度 → PCL 物件 → 同一個 MTC，
  完整 Pick and Place 規劃與 CLI 節點／控制器／服務／Action 檢查；
  `/tmp/kinova-ai-drink-can-plan.log`、`.cli`、`-result.log`。
  `drink can` confidence=0.615，選中 cylinder_0，depth_overlap=1.000，保留 6 個碰撞物件；
  MTC 更新目標尺寸為 `[0.100601,0.019040]`，並輸出 `Task planning succeeded`。
- Passed：`scene_source:=known perception_method:=ai` 在啟動節點前拒絕，
  `/tmp/robot-ai-invalid-launch.log`；獨立 `object_detect.py --help` 與 `git diff --check`。
- Passed：真實 AI 服務以 `ai_confidence:=0.99` 拒絕飲料罐，回傳 success=false，
  MTC 明確退出 code 1，沒有開始規劃或退回幾何目標；
  `/tmp/kinova-ai-reject.log`、`-result.log`。ROS launch 在 Shutdown 事件後本身返回 0，
  因此以 MTC 子程序狀態與服務結果判定；關閉時仍有既有 MoveIt exit -11。

使用實際模型 `yoloe-26n-seg.pt`、CPU PyTorch 2.14.1、Ultralytics 8.4.178、NumPy 1.26.4。
AI 依賴位於 `.venv`，權重／文字編碼器位於 `~/.cache/robot_perception/yoloe`。
初次 AI 啟動曾因回呼命名覆蓋 rclpy Node.handle 而失敗，已改名 handle_scene，
上述成功測試為修正後的重跑結果。

提示詞影響零樣本辨識：同一張 Gazebo 圖片上 `can` 沒有結果，`drink can` 約 0.61、
`soda can` 約 0.72、`cube` 約 0.61、`apple` 約 0.18，`orange` 沒有結果。
這些僅是單張辨識結果，不等同水果／紙盒已能完成抓取規劃；目前完整 AI 規劃驗證對象為飲料罐。
示意圖 `/tmp/kinova-ai-prompt-drink-can.png`，文字結果 `/tmp/robot-ai-prompts-check.log`。

本次未更動使用者的抓取角度或任何 MTC stage，所有完整流程測試均 `execute=false`。
**未驗證 AI 模式的 Gazebo 軌跡執行或物理夾持，也未重測 CUDA／GUI。**
既有 MoveIt 2.12.4 執行／關閉崩潰未在此辨識整合中修正，詳見後續既有紀錄。

重現成功測試（先 source ROS、workspace、`.venv/bin/activate`）：

```bash
python3 tests/run_demo_check.py --robot kinova_gen3_lite --scene-source perception \
  --perception-method ai --ai-target-class "drink can" --timeout 240 \
  --log /tmp/kinova-ai-drink-can-plan.log
```

## 抓罐失敗：夾爪逾時與另一個 MoveIt 執行期崩潰

使用者 13:39:52 啟動（move_group PID 433322）的首個任務失敗為：

```text
gripper_controller_handle: waitForExecution timed out
Controller is taking too long to execute trajectory (the expected upper bound for the trajectory execution was 2.443334 seconds).
Completed trajectory execution with status TIMED_OUT
```

MTC 隨後報 execution code 99999，Launch 才送 SIGINT 關閉各程序。該次 move_group exit -11
及 Gazebo 被 SIGTERM 終止（exit -15）發生於清理階段，不能代替首個夾爪錯誤來診斷。

診斷測試／設定：

- 未調參數的無視窗抓罐測試曾完整執行成功；`/tmp/kinova-can-execution-headless-before.log`
  與 `-result.log`。關節監看 `/tmp/kinova-can-headless-joints.log` 顯示閉合目標 0.7rad 時，
  實際約 0.636rad 後停止，未到位置容差內但控制器允許 stall 成功。這只證明該次回報完成，
  不能單憑關節位置認定已穩定夾住物件，原使用者失敗那次沒有保存相同關節資料。
- 現僅對 Kinova 夾爪 mapping 設 allowed_goal_duration_margin=3.0 秒，增加接觸後穩定與
  stall 判定的有界餘裕。未放寬手臂誤差、修改碰撞檢查或停用 duration monitoring。
- Passed：整合套件增量建置，`/tmp/kinova-can-timeout-build.log`；16 項 Profile 檢查。
- Passed（單次無視窗重測）：修改後的 perception + 抓罐 + execute=true 完整軌跡及 CLI 檢查，
  `/tmp/kinova-can-gripper-margin-headless.log`、`.cli` 與 `-result.log`，包含 Task executed successfully。
- GUI 重測兩次另發現 move_group 在任務執行中（尚未發送控制器 goal）崩潰：
  PlanExecution::isRemainingPathValid → successfulTrajectorySegmentExecution，位址 0x58。
  日誌 `/tmp/kinova-can-execution-before.log` 與 `/tmp/kinova-can-gripper-margin.log`；
  兩次都不算通過。專用測試器已停止並清理自己的模擬。

此執行期崩潰尚未修正，不能再把所有 MoveIt exit -11 一律歸類為關閉問題，
也不能宣稱 GUI 抓罐已穩定修復。套件庫目前安裝與候選版本皆為 MoveIt 2.12.4。

參數及完成判定對照官方原碼：
[MoveIt 2.12.4 Controller deadline](https://github.com/moveit/moveit2/blob/2.12.4/moveit_ros/planning/trajectory_execution_manager/src/trajectory_execution_manager.cpp)、
[Jazzy GripperActionController stall 判定](https://github.com/ros-controls/ros2_controllers/blob/jazzy/gripper_controllers/include/gripper_controllers/gripper_action_controller_impl.hpp)。

## 生活物品移近手臂：位置與規劃 Passed

保留原長圓柱 `[0.35,0.15,0.175]` 與放置點，只調整四個生活物品的平面位置：

| 物品 | 新 xyz（m） | 距底座水平距離（舊→新） |
|---|---|---|
| 橘子 | `[0.24,0.28,0.026]` | 60→37cm |
| 蘋果 | `[0.40,0.02,0.025]` | 60→40cm |
| 飲料罐 | `[0.30,-0.02,0.05]` | 55→30cm |
| 紙盒 | `[0.28,-0.30,0.0325]` | 61→41cm |

同步更新 kinova_grasp_can.yaml 的已知目標位姿與 Kinova README，未修改尺寸、質量或共用任務。

- Passed：SDF Valid、兩個整合套件增量建置；`/tmp/kinova-nearer-props-build.log`。
- Passed：靜態比對四物品均比原位置接近底座、長圓柱位姿不變、飲料罐 YAML 與 world 一致。
- Passed：感知產生 6 個碰撞物件，原長圓柱與新位置的飲料罐均完整規劃成功。
- Passed：飲料罐測試的 CLI／Controller／服務／Action 與核心節點檢查。
- 限制：原長圓柱測試的單次 CLI 快照漏列 controller_manager 等 Gazebo 節點，測試器斷言失敗；
  該次日誌的 Task planning succeeded 有保留，不將整個測試器退出狀態宣稱為成功。
- 未測：本次 execute=false，未重跑實際軌跡；水果與紙盒未逐個驗證抓取可達性或物理夾持。

日誌：`/tmp/kinova-nearer-props-default.log`、`/tmp/kinova-nearer-props-can.log`，
各自的 `.cli` 與 `-result.log`。

## Controller readiness 查詢卡住修正：Passed

使用者 13:29:27 的 GUI 啟動紀錄顯示，三個 spawner 在 13:29:36 前皆已成功 activated，
readiness 仍每 5 秒列出全部控制器未 active，其他 JointState／TF／clock／感知條件均已通過。
舊程式對 ListControllers 的單筆 future 沒有逾時重試，未收到回覆時會一直等待到全域 timeout；
原始服務回覆未送達的底層原因無法由當時日誌判定。

readiness.py 現對單筆查詢設定 3 秒 wall-time 上限，移除已逾時的 pending request 後重試，
並記錄服務名稱與重試原因。仍須從回覆確認所有控制器 active，才允許啟動 MTC；
未加入固定睡眠或跳過啟動檢查，也未改動物品場景或任務核心。

- Passed：robot_bringup 增量建置；`/tmp/readiness-retry-build.log`。
- Passed：真實 ROS mock service 刻意保留第一筆回覆，3 秒後重試成功，第一筆仍未回覆時即可正確就緒。
- Passed：mock controller 維持 inactive，readiness 必須非零退出，確認沒有繞過檢查。
- Passed：Kinova GUI + RViz + 新物品場景 + perception + execute=true，完整 MTC 規劃、執行與 CLI 檢查。
- Passed：共用 readiness 的 myCobot known 規劃回歸；`/tmp/mycobot-readiness-retry-regression.log`、`.cli` 與 `-result.log`。
- 既有 move_group 關閉時的 exit -11 仍存在；本次實際任務先成功，崩潰發生在測試清理階段。

回歸測試：`python3 tests/check_readiness_retry.py`，日誌 `/tmp/readiness-retry-regression.log`。
GUI 全流程：

```bash
python3 tests/run_demo_check.py --robot kinova_gen3_lite --scene-source perception \
  --with-gui --execute --timeout 300 --log /tmp/kinova-gui-readiness-fixed.log
```

GUI 日誌為上述路徑、`.cli` 與 `/tmp/kinova-gui-readiness-fixed-result.log`，包含
`Task executed successfully` 及 `Passed: complete demo plan and execution`。

## Kinova 水果／生活物品場景：Passed

在原場景增加簡化蘋果（直徑 50mm）、橘子（52mm）、飲料罐（直徑 40mm、高 100mm）、
小紙盒（45×35×65mm）。均為獨立動態模型，具有有效的質量、慣性與碰撞形狀，無外部網格依賴。
保留既有長圓柱與放置位置，未修改 MTC、共用感知演算法或 myCobot 場景。

- Passed：`gz sdf -k`，SDF Valid。
- Passed：kinova_gen3_lite_support、robot_bringup 增量建置；`/tmp/kinova-household-build.log`。
- Passed：新場景實際載入，RGB／點雲就緒，感知產生 6 個碰撞物件（含支撐面）。
- Passed：預設長圓柱仍完整規劃成功；`/tmp/kinova-household-default.log`、`.cli` 與 `-result.log`。
- Passed：使用 kinova_grasp_can.yaml，實測選中高約 0.1006m、半徑約 0.019m 的飲料罐，
  完整 Pick and Place 規劃成功；`/tmp/kinova-household-can.log`、`.cli` 與 `-result.log`。
- Passed：兩次測試的 CLI 控制器／服務／Action 與核心節點唯一性檢查、git diff --check。
- 未驗證：新物品的物理夾持／搬運／放置；水果與紙盒的抓取規劃也未測。

水果是球形外觀與碰撞模型，現有感知仍以 cylinder／box 幾何近似，不宣稱水果種類辨識。
新增目標只透過既有 task_params_file 參數覆寫；預設啟動仍抓原長圓柱。

```bash
python3 tests/run_demo_check.py --robot kinova_gen3_lite --scene-source perception \
  --task-params-file src/robot_pick_place/robot_bringup/config/scenes/kinova_grasp_can.yaml \
  --timeout 180 --log /tmp/kinova-household-can.log
```

## Kinova execute=true 後續修正：完整軌跡執行 Passed

使用者日誌中的首個執行錯誤不是關閉階段的 segmentation fault，而是：

```text
Could not find JointModelGroup that actuates {joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, right_finger_bottom_joint}
```

Kinova SRDF 原本只有獨立的 arm 與 gripper；現新增 arm_with_gripper subgroup 聯集，
讓 MTC executor 可以表示 Connect 合併後的七關節軌跡。未修改通用 MTC、模型或 myCobot。

第一次執行重測開始移動後，接續發現 JTC 拒收 Cartesian stage：
`Velocity of last trajectory point of joint joint_1 is not zero: 0.000718679869092`。
Kinova 模擬控制器現比照原 myCobot 設定，使用 allow_nonzero_velocity_at_trajectory_end=true。
沒有放寬起始位置、路徑／終點誤差或移除碰撞檢查。

- Passed：Kinova 整合套件重新建置；`/tmp/kinova-execution-group-build.log`。
- Passed：新增合併群組回歸檢查，Profile 測試共 16 項。
- Passed：perception + execute=true，非空相機資料、感知場景、完整規劃與完整軌跡執行。
- Passed：控制器、服務／Action、核心節點唯一性檢查。
- 未驗證：Gazebo 物件的穩定夾持與最終放置位姿；控制軌跡成功不等同物理物件搬運成功。
- 既有問題：成功後送出 SIGINT 清理時，已安裝的 move_group 仍出現 exit -11。

```bash
python3 tests/run_demo_check.py --robot kinova_gen3_lite \
  --scene-source perception --execute --timeout 300 \
  --log /tmp/kinova-execute-controller-fixed.log
```

最終日誌 `/tmp/kinova-execute-controller-fixed.log`、`.cli` 與
`/tmp/kinova-execute-controller-fixed-result.log`：

```text
Task planning succeeded
Executing the planned task
Task executed successfully
Passed: CLI controllers/topics/services/actions and unique core node counts
Passed: complete demo plan and execution
```

群組修正後首次重測（尚未修正控制器末速）的紀錄保留在
`/tmp/kinova-execute-group-fixed.log`，不將該次失敗算作成功。

## Kinova Gen3 Lite（6DOF、Gen3 Lite 2F）：Passed

官方來源：ros2_kortex 的 Jazzy commit `462dab9aa4732d733be55e1846530dd920c7c7d3`。
使用現有 robot_bringup Profile 與同一個 robot_task_core executable；核心 C++、感知與 myCobot
程式本次均未修改。官方模型僅有記錄於 patch 的兩行 Gazebo 介面修正，未更動幾何／慣性／限制。

| 驗證項目 | 結果與證據 |
|---|---|
| 全工作區建置 | Passed：21 套件；`/tmp/kinova-full-build.log` |
| 新整合依賴檢查 | Passed：rosdep check All system dependencies have been satisfied |
| 原 Profile 12 項＋Kinova 3 項 | Passed：15 項，涵蓋六軸、夾爪 mimic、控制關節不重複、正確硬體 plugin、SRDF 姿態、URDF→SDF 相機保留 |
| Gazebo 模型 | Passed：官方六軸與 2F 網格可載入，兩個 GazeboSimSystem 共用單一 Controller Manager；無 fake／實體硬體 plugin |
| Controller | Passed：joint_state_broadcaster、joint_trajectory_controller、gen3_lite_2f_gripper_controller 均 active |
| Controller 實際動作 | Passed：夾爪 0→0.35→0 的 GripperCommand、手臂 joint_1 0→0.1→0 的六關節 FollowJointTrajectory，Action 成功且 JointState 回饋吻合 |
| Joint State、TF、clock、MoveIt | Passed：readiness 與 CLI 檢查，各核心節點僅一份；ExecuteTaskSolution Action 存在 |
| known 完整 MTC 規劃 | Passed：`/tmp/kinova-known-aligned.log`、`.cli`、`/tmp/kinova-known-aligned-result.log` |
| perception 完整 MTC 規劃 | Passed：RGB／點雲非空，辨識目標圓柱與支撐面，共 2 物件，完整 Task planning succeeded；`/tmp/kinova-perception-camera-fixed.log`、`.cli` 與 `-result.log` |
| myCobot 感知與完整規劃回歸 | Passed：7 個物件、完整 MTC 解、CLI 與核心節點檢查；`/tmp/mycobot-after-kinova-regression.log`、`.cli` 與 `-result.log` |
| Kinova 完整軌跡執行／物理抓放 | Not Tested：完整任務測試均 execute=false，未宣稱夾持或搬運成功 |

Action 動作額外測試日誌：`/tmp/kinova-controller-actions.log`、`.cli` 與
`/tmp/kinova-controller-actions-result.log`。這項測試僅在自己的隔離模擬送出簡單控制目標，
不是整個 MTC 抓放軌跡的實際執行。測試後回到初始姿態。

兩種 Kinova 模式最後均輸出：

```text
Task planning succeeded
Execution skipped as per configuration
Passed: CLI controllers/topics/services/actions and unique core node counts
Passed: complete demo plan; execute=false skipped execution
```

規劃測試保留完整 Pick、Attach、Lift、Move to Place、Place、Detach、Retreat、Home。
場景目標圓柱位於 `[0.35,0.15,0.175]`、放置底面 `[0.35,-0.15,0]`。
沒有透過關閉整體碰撞檢查來取得成功解。

整合過程曾發現：抓取方向沿用 myCobot 會讓 Kinova 手指與圓柱碰撞；現已只在 Kinova YAML
調整抓取 yaw。外部相機原本的空 link 在 SDF 轉換時消失；已加上相機慣性並固定至 base_link，
實測已收到 RGB／點雲，新增的 SDF 轉換檢查也能覆蓋這個問題。

既有環境限制仍存在：SIGINT 關閉時已安裝的 move_group 有 Executor 解構崩潰；
Gazebo DART 不支援原生 mimic constraint，gz_ros2_control 仍讀取並處理 URDF mimic；
GripperActionController 有 deprecated 提示。這些不是本次完整規劃的失敗，物理夾持仍須另驗證。

## 資料夾整理

後續將四個 `bme_ros2_tutorials_*` 套件集中到 `src/bme_ros2_tutorials/`，
與 `src/robot_pick_place/` 並列。四個教學套件重新建置成功，搬移前後仍可探索相同的
18 個套件，套件查找與安裝連結檢查通過。教學程式內容不變，README 目錄圖已更新。
建置日誌：`/tmp/tutorial-layout-build.log`。

依使用者要求，將 `mycobot_ros2` 與四個 `robot_*` 套件集中到
`src/robot_pick_place/`。這一層僅為來源分類，不新增 ROS 套件或 workspace。
同步更新 README、文件與來源保留測試的檔案路徑；未修改抓放或感知邏輯。

- Passed：搬移前後 colcon 探索的 18 個套件名稱一致。
- Passed：清除舊路徑快取後，全工作區 18 個套件重新建置成功（4 分 39 秒）；有既有 CMake 依賴警告，無建置失敗。
- Passed：12 項 Profile、模型／控制器資源與來源保留測試。
- Passed：新位置安裝後的統一 Launch 可列出參數。
- Passed：安裝連結檢查，沒有失效連結或指向舊來源位置的連結。
- 本次僅驗證搬移與建置，未重跑 Gazebo 抓放。

舊建置／安裝產物備份在 `log/layout-generated-backup-6fjencmg/`；重新建置日誌為
`/tmp/robot-layout-build.log`。使用者先前已回報 Gazebo 實際抓放成功，屬使用者驗證，
與下方當時 `execute=false` 的自動測試紀錄區分。

## 原本感知流程接入共用任務：Passed

新入口預設 `scene_source=perception`，沿用 Gazebo → 相機／點雲 → 感知場景 → MTC。
`known` 僅為可選測試。本次沒有新增機器人、修改 PCL 演算法或擴充 MTC Stage。

第一次實測已套用 7 個物件，但所有抓取解被 `torso_link colliding with camera_head_link`
拒絕。URDF 確認兩者透過固定 `camera_head_joint` 相鄰，安裝幾何重疊；因此在 myCobot
SRDF 補上這一對 Adjacent 排除，沒有關閉手臂與相機／環境的碰撞檢查。

- Passed：mycobot_moveit_config、robot_bringup 增量建置。
- Passed：12 項 Profile/來源保留檢查、`git diff --check`。
- Passed：Launch 參數列出的預設場景為 perception。
- Passed：Gazebo 相機非空 RGB／點雲、感知服務、7 個物件套用至 MoveIt。
- Passed：完整 MTC Pick and Place 規劃；`execute=false` 確認跳過軌跡執行。
- Failed（測試器 CLI 快照）：規劃成功後的單次 `ros2 node list --no-daemon` 未列出
  move_group 與感知節點，觸發唯一節點斷言；日誌確認兩者在快照後的測試清理階段才收到 SIGINT。
  因此本次不宣稱整個測試器退出成功，也不宣稱 CLI 節點數檢查通過。
- Not Tested：Gazebo 實際抓取／搬運／放置及物理夾持。

修正前日誌：`/tmp/robot-original-flow-check.log`；修正後：
`/tmp/robot-original-flow-fixed.log` 與 `.cli`；建置：`/tmp/robot-original-flow-build.log`。

```text
[Scene] Applied 7 collision objects; target='cylinder_1', support='support_surface'
Task planning succeeded
Execution skipped as per configuration
```

## 先前已知物件完整任務規劃：Passed

實際測試啟動等效於：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_bringup demo.launch.py \
  robot:=mycobot_280 scene_source:=known execute:=false \
  headless:=true use_rviz:=false
```

由 `tests/run_demo_check.py` 在獨立 ROS domain / Gazebo partition 啟動，
設定等待上限，結束後僅清理自己建立的程序。
完整日誌：`/tmp/robot-pick-place-verified.log`；CLI：同路徑加 `.cli`；
測試摘要：`/tmp/robot-pick-place-verified-result.log`。

最後一次實際輸出：

```text
[Place IK] Target object center in base_link: [-0.1830, -0.1400, 0.1750]
Task initialized successfully
Task planning succeeded
Execution skipped as per configuration

Passed: CLI controllers/topics/services/actions and unique core node counts
Passed: complete demo plan; execute=false skipped execution
```

這是整個 MTC Task 的成功解，包含 Pick、Attach、Lift、Move to Place、Place、Detach、
Retreat 與 Home；並非只有單獨的 Place IK 成功。原放置 xyz/rpy 未修改。

CLI 確認：

```text
gripper_action_controller  position_controllers/GripperActionController          active
arm_controller             joint_trajectory_controller/JointTrajectoryController active
joint_state_broadcaster    joint_state_broadcaster/JointStateBroadcaster           active
```

`/move_group`、`/controller_manager`、`/robot_state_publisher` 各一份；known 中沒有
`/robot_perception_server`。readiness 已確認 Joint States、base→IK Frame TF 與 clock。
ROS-GZ bridge 在 known 模式仍可能列出 lazy camera topic 名稱，但 URDF 沒有啟動 sensor，
亦未啟動 image_bridge 或感知 Server。

## Place IK 修正範圍

- 使用附著物件 Frame 作為 Place IK 的目標 Frame，保留實際抽樣後的抓取關係。
- 高度投影使用明確的 `Eigen::Matrix3d` 值；原 `auto` 表達式參考到暫存 Isometry，
  是本次修改引入的生命週期錯誤，已修正。
- 圓柱中心目標保持 `[-0.183,-0.14,0.175]`，沒有更換成較容易的放置點。
- 先前已修正 SRDF Group 未包含所有 mimic 手指時的接觸清單；允許接觸與恢復檢查使用同一組實際 Link。
- 保留 Stage 順序、規劃器原理、URDF/SRDF 幾何、關節限制及開合命名姿態。

## 驗收狀態

| 項目 | 狀態 | 證據與界線 |
|---|---|---|
| 原專案基準建置 | Passed | 5 個相關套件；`/tmp/robot-baseline-build.log` |
| 全 workspace 建置 | Passed | `colcon build --symlink-install --parallel-workers 2`，18 packages；`/tmp/robot-full-build.log` |
| 最終修正編譯 | Passed | 新模組增量建置 3 packages 通過；最後 Place 高度修正的 robot_task_core 另建置通過；`/tmp/robot-launch-override-build.log`、`/tmp/robot-place-height-build.log` |
| rosdep 依賴檢查 | Passed | `rosdep check --from-paths src --ignore-src`：All system dependencies have been satisfied |
| 四個新 Package 安裝及 Launch 介面 | Passed | `ros2 pkg prefix`、`ros2 launch robot_bringup demo.launch.py --show-args` |
| 通用套件不依賴 mycobot_* | Passed | manifest/CMake/include/Launch 檢查；test_profiles.py |
| 通用 Service Type 生成與 Client/Server 型別相容 | Passed | 真實 round trip；缺少 sensor 時 success=false；`/tmp/robot-phase1-test.log` |
| Profile 解析、錯誤欄位與資源驗證 | Passed | `tests/test_profiles.py` 共 12 項，包括未知 robot/package、Group/Frame/Controller、替代 install prefix |
| PCL helpers 與 MTC Stage 順序保留 | Passed | 與舊來源逐一比對，包含於上述 12 項檢查 |
| 錯誤處理與等待上限 | Passed | `tests/check_task_safety.py` 共 9 項；`/tmp/robot-phase2-test.log` |
| known 完整 MTC Pick and Place 規劃 | **Passed** | 最後一次完整 Task planning succeeded，保留原放置位置 |
| known 不等待/啟動感知 | Passed | 無 perception client、無感知 Server、無 sensor；真實執行與靜態檢查 |
| 共用 Robot Description、SRDF、Controller 設定 | Passed | 同一份 URDF 原文與設定傳給 RSP/MoveIt/MTC；runtime plugin 路徑及 Controller 關節對應已檢查 |
| Gazebo 模型、Controller、Joint States、TF | Passed | 無 GUI 啟動、readiness 與 CLI 通過 |
| 不重複啟動核心節點 | Passed | 最後一次 CLI 實際各一份 move_group / controller_manager / RSP |
| execute=false | Passed | 完整規劃後明確跳過 `task.execute()` |
| Gazebo 完整軌跡執行、實際物理夾持 | **Not Tested** | 沒有執行 execute=true；MoveIt Attach 不代表物理抓住 |
| 感知相機、RGB、PointCloud、通用 Server 啟動 | Passed | 曾達到 perception readiness：非空點雲/RGB 與服務皆可用；`/tmp/robot-perception-full.log` |
| 實際點雲 Service → Scene → 完整 MTC 規劃 | **Not Tested（最終版本）** | 前次在參數覆寫問題處停止；已修正 YAML 覆寫，但依使用者最後指示未擴大重跑感知整合 |
| 舊 Package 與入口保留 | Passed（建置/來源） | 舊 Package 全數可建置、原來源與 Service Type 保留；Gazebo/MoveIt 入口使用同套實際啟動路徑 |
| 舊四終端感知 Demo 全流程回歸 | **Not Tested** | 未宣稱舊 MTC 已完成整套實際執行；保留來源作對照 |
| 未來機器人不需複製 Stage 的契約 | Passed（架構） | 新 Profile/Launch/必要 adapter 提供參數，通用任務無型號分支；其他機器人尚未整合驗證 |
| 執行後原始 Controller YAML 不被寫回 | Passed | runtime 檔案另存暫存目錄；指定 ros2_controllers.yaml 的 git status 無變更 |
| git diff 格式檢查 | Passed | `git diff --check` |

安全測試九項為：非法 source、Group、Frame、Pose、尺寸、支撐面 ID、known 無 perception、
perception availability timeout、perception success=false。錯誤均停止規劃並非零退出。

## 已知環境問題

沙箱原先阻擋本機 ROS/Gazebo socket 與預設 ROS_HOME 寫入；完整動態測試改用獲准的本機
執行、獨立 domain/partition，以及 /tmp 下的測試 ROS_HOME/日誌。

完整規劃成功後，測試以 SIGINT 關閉 Launch 時，環境內已安裝的
`/opt/ros/jazzy/lib/moveit_ros_move_group/move_group` 出現 `Executor` 解構時的
segmentation fault（exit -11）。原因尚未追查，未將其冒稱為乾淨關閉。
測試器已限定清理本次 partition 的程序，沒有使用廣泛 pkill。

此關閉問題不改變最後一次完整規劃成功與 execute=false 的證據；依使用者要求，本次不再
延伸處理安裝套件的關閉行為，也不進一步驗證物理執行或完整感知流程。

## RViz 空白畫面修正（同日後續）

已重現：Gazebo/RSP 的 rviz_config_file 預設值洩漏到 MoveIt Launch，形成
`<moveit_share>/rviz/<description_share 的絕對路徑>`，該檔案不存在。
在 demo.launch.py 以 scoped GroupAction 隔離兩個廠商 include；控制器的延後 spawner
則在 OpaqueFunction 中先解析並保存 manager/timeout/use_sim_time，避免 scope 結束後失去參數。
未修改 MTC Stage、場景或機器人模型。

- Passed：mycobot_moveit_config、robot_bringup 增量建置。
- Passed：12 項 Profile/來源保留檢查。
- Passed：實際 RViz 命令指向存在的 mycobot_moveit_config/rviz/move_group.rviz。
- Passed：RViz 日誌已載入 mycobot_280 RobotModel，MotionPlanning 已連上 arm group。
- Passed：啟用 RViz 的完整 known/execute=false 規劃、Controller/CLI 與核心節點數檢查。
- 限制：WSLg 的視窗截圖取得黑色影像，未據此聲稱完成肉眼視覺確認。

本次日誌：`/tmp/robot-rviz-verified.log`、`/tmp/robot-rviz-verified-result.log`；
建置：`/tmp/robot-rviz-fix-build.log`。
