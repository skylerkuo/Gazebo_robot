# Kinova Gen3 Lite：ROS 2 Jazzy / Gazebo Harmonic

使用 [Kinova 官方 ros2_kortex Jazzy 分支](https://github.com/Kinovarobotics/ros2_kortex/tree/jazzy)，
固定 commit `462dab9aa4732d733be55e1846530dd920c7c7d3`（版本 0.2.6）。
來源位於相鄰的 `ros2_kortex/`，版本記錄於 `upstream.repos`。未使用 Gazebo Classic。
只接入六軸 Gen3 Lite 與 Gen3 Lite 2F，沒有建立 Gen3／Robotiq 的 Profile。

## 建置與啟動

在 workspace 根目錄執行（目前來源已下載、修正並建置）：

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 colcon build --symlink-install --parallel-workers 2
source install/setup.bash

# 外部 RGBD 相機 → 共用感知 → 共用 MTC，僅規劃
ros2 launch robot_bringup demo.launch.py robot:=kinova_gen3_lite execute:=false

# 已知物件模式（不啟動相機／感知）
ros2 launch robot_bringup demo.launch.py robot:=kinova_gen3_lite scene_source:=known execute:=false
```

無 GUI 加上 `headless:=true use_rviz:=false`。要送出規劃軌跡可用 `execute:=true`；
初次整合驗證採 execute=false；後續已驗證 perception + execute=true 的完整 MTC 軌跡執行，
日誌顯示 Task executed successfully。後續終端模式已量測實際罐子位移，詳見工作區 docs/validation.md。
請與 myCobot 擇一啟動，兩個 Profile 使用相同的 ROS topic／service／controller manager 名稱。

## 整合位置

- Profile：`robot_bringup/config/robots/kinova_gen3_lite.yaml`。
- 場景與感知：`robot_bringup/config/scenes/kinova_{known,perception,perception_settings}.yaml`。
- 本套件 `urdf/gen3_lite.urdf.xacro` 呼叫官方 `load_robot`，保留官方機構、網格、慣性、
  關節限制及 URDF mimic 關係；加上 Harmonic plugin 與可選的外部相機。不是 Gen3 內建視覺模組。
- `config/` 的 MoveIt 限制、規劃設定與 controller mapping 取自官方 Lite 設定。
  SRDF 修正 end_effector 的 group 為 gripper、指定 parent_group=arm、移除與 mimic 重複的
  passive_joint，並新增 home/open/closed 命名姿態；原碰撞排除表保留。
  arm_with_gripper 包含 arm 與 gripper，供 MTC executor 識別 Connect 合併的七關節軌跡。
- `config/ros2_controllers.yaml` 僅啟動 JointStateBroadcaster、六軸 JointTrajectoryController、
  Gen3 Lite 2F GripperActionController，不載入實體通訊／twist／fault controller。
  MoveIt 的夾爪 Controller mapping 設定 allowed_goal_duration_margin=3.0 秒，
  留給接觸後的停滯完成判定；手臂與其他執行逾時檢查保留。
  模擬軌跡控制器沿用 myCobot 的 allow_nonzero_velocity_at_trajectory_end=true，
  接受 MTC Cartesian stage 末端的小量非零速度。
- `launch/` 滿足現有 bringup 契約，直接使用準備好的同一份 URDF、Controller YAML 與 MoveIt
  字典。只有一份 robot_state_publisher、一個 gz_ros2_control controller manager、一個 move_group。
- 抓取 TCP 相對 end_effector_link 沿局部 Z 軸偏移 +0.13m，抓取方向依 Lite 的手指閉合方向設定；
  預設飲料罐中心 `[0.30,-0.02,0.05]`；投放時物品底面到 `[0.25,-0.52,0.18]`。
- 仍執行 `robot_task_core/mtc_node`。後續互動與收集箱功能沿用相同任務，未複製 MTC。

## 水果與生活物品場景

`worlds/pick_place.sdf` 已移除蘋果、紅色長圓柱與紙盒，目前包含以下不需下載外部網格的物品：

| 物件 | 尺寸 | 位置 xyz（m） |
|---|---|---|
| 橘子 | 直徑 52mm | 0.24, 0.28, 0.026 |
| 黃綠色網球（白色接縫） | 直徑 52mm | 0.40, 0.02, 0.026 |
| 綠色瓶子（瓶頸、藍蓋、淺色標籤） | 主體直徑 36mm、總高 120mm | 0.35, 0.15, 0.06 |
| 藍色飲料罐 | 直徑 40mm、高 100mm | 0.30, -0.02, 0.05 |
| 紅色馬克杯（深色杯口、把手） | 杯身直徑 44mm、高 70mm | 0.28, -0.25, 0.035 |

每個物件都是獨立動態模型，包含碰撞形狀、質量及慣性；水果使用球形與果梗呈現外觀。
生活物品距底座的水平距離約 30～41cm，分散擺放。
目前 PCL 仍以 cylinder／box 近似物件；選用 AI 可透過英文文字辨識目標。
原本啟動指令會載入新場景，known／幾何模式的預設目標已同步改為藍色飲料罐。

AI 終端建議輸入 `green plastic bottle`、`tennis ball`、`drink can`。
同一張實際相機畫面上，前兩者的 YOLOE 信心分別約 0.616／0.387，`drink can` 約 0.817。
這是該畫面的量測，不保證所有視角與擺放皆相同；泛稱 `bottle` 可能選到藍色飲料罐。
紙盒已換成紅色馬克杯。AI 建議輸入 `red cup`（實際相機單張信心約 0.251），
`red mug` 也能選中（約 0.147）；泛稱 `cup`／`mug` 在該畫面未檢出。
杯身採圓柱碰撞與對應慣性，把手與深色杯口僅呈現外觀；目前抓杯身，不支援抓把手或盛裝液體。
瓶子的視覺包含瓶頸與瓶蓋，物理碰撞與慣性採整體圓柱包覆，與目前 primitive 抓取近似一致。

可透過現有 task_params_file 改為飲料罐，不需要修改 MTC 或 Profile：

```bash
ros2 launch robot_bringup demo.launch.py robot:=kinova_gen3_lite \
  task_params_file:="$(ros2 pkg prefix --share robot_bringup)/config/scenes/kinova_grasp_can.yaml" \
  execute:=false
```

收集箱為開口模型，外尺寸 32×22×10cm，底面中心 `[0.25,-0.52,0]`；
具有箱底與四片箱壁的真實碰撞形狀，MoveIt 以相同五片 BOX 建模。
投放區在目前點雲 crop 之外，避免重複選到箱內物品。預設物品底面在箱緣上方 8cm 開夾，
跳過向下放置階段；箱內物品由物理模擬落下，不逐一規劃箱底位置。
設定位於 bringup 的 kinova_known.yaml、kinova_perception.yaml，使用 drop_into_bin=true。
各物品實際抓放結果見工作區 docs/validation.md。

## 官方來源的最小相容修正

`patches/ros2_kortex-jazzy-lite-gazebo.patch` 只有兩行有效修改：

1. 將 Lite 夾爪使用的 `gz_ros2_control/GzSimSystem` 改為 Jazzy 已安裝 plugin 匯出的
   `gz_ros2_control/GazeboSimSystem`。
2. Gazebo 模式的 right_finger_bottom_joint 只由 GripperHardwareInterface 擁有，避免在 arm
   hardware 重複註冊同一個 joint。實體模式保留原本行為。

URDF 的 mimic multiplier／offset 維持官方值，gz_ros2_control 實測從 URDF 讀取這些關係。
Gazebo DART 仍可能記錄不支援原生 mimic constraint 的警告；不因此宣稱已驗證物理夾持。
沿用 GripperCommand 與 named_pose 即可整合，沒有另寫夾爪 adapter。

工作區以 COLCON_IGNORE 排除官方 kortex_api、kortex_driver、kortex_bringup 及另外兩款
Gen3 MoveIt config；本次模擬不需要下載二進位硬體 API，也不接實體手臂。

若在另一個乾淨 checkout 重新取得官方來源：

```bash
vcs import src/robot_pick_place --skip-existing \
  --input src/robot_pick_place/kinova_gen3_lite_support/upstream.repos
git -C src/robot_pick_place/ros2_kortex apply \
  ../kinova_gen3_lite_support/patches/ros2_kortex-jazzy-lite-gazebo.patch
for package in kortex_api kortex_driver kortex_bringup \
  kortex_moveit_config/kinova_gen3_6dof_robotiq_2f_85_moveit_config \
  kortex_moveit_config/kinova_gen3_7dof_robotiq_2f_85_moveit_config; do
  touch "src/robot_pick_place/ros2_kortex/$package/COLCON_IGNORE"
done
```

目前工作區已套用 patch，勿重複套用；可用 `git diff` 在官方來源內檢查兩項修改。

## 驗證

```bash
python3 -m unittest discover -s tests -p 'test*profile*.py'
python3 tests/run_demo_check.py --robot kinova_gen3_lite --scene-source known
python3 tests/run_demo_check.py --robot kinova_gen3_lite --scene-source perception --timeout 180
# 在自己的隔離 Gazebo 中規劃並執行整個 MTC 軌跡
python3 tests/run_demo_check.py --robot kinova_gen3_lite --scene-source perception --execute --timeout 300
```

測試器使用獨立 domain／partition，只清理自己建立的程序。控制器可另用
`tests/check_kinova_controllers.py` 對專用測試模擬送出夾爪開合與手臂往返目標；
此腳本會移動模擬手臂，請勿用於正在執行其他任務的節點。

實際測試結果與日誌見 workspace 的 `docs/validation.md`。
抓罐的 GUI 測試另觀察到 MoveIt 2.12.4 在 PlanExecution::isRemainingPathValid 執行中崩潰，
尚未修正；請勿把曾經的軌跡成功紀錄解讀為目前所有模式均穩定或物理夾持已驗證。
