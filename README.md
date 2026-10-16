# Gazebo_robot：共用 Pick and Place 任務

目前整合 **myCobot 280** 與 **Kinova Gen3 Lite（6DOF、Gen3 Lite 2F）**。共用 `robot_task_interfaces`、`robot_perception`、
`robot_task_core`、`robot_bringup`，由 Robot Profile 提供機器人設定；原有
`src/robot_pick_place/mycobot_ros2/` 套件與入口保留。

原始碼整理在同一個 `src/robot_pick_place/` 資料夾：

```text
ros2_ws/
├── src/
│   ├── robot_pick_place/       # 抓放專案，這一層只是整理資料夾
│   │   ├── mycobot_ros2/       # myCobot 專用套件與舊 Demo
│   │   ├── ros2_kortex/        # 官方 Jazzy 來源（固定版本，僅建置 Lite 所需套件）
│   │   ├── kinova_gen3_lite_support/ # Kinova 模型整合、MoveIt／Controller／Launch
│   │   ├── robot_bringup/      # 共用啟動與各手臂設定
│   │   ├── robot_task_core/    # 共用 MTC 任務
│   │   ├── robot_perception/   # 共用感知
│   │   └── robot_task_interfaces/ # 共用服務格式
│   └── bme_ros2_tutorials/    # 教學專案
│       ├── bme_ros2_tutorials_bringup/
│       ├── bme_ros2_tutorials_cpp/
│       ├── bme_ros2_tutorials_interfaces/
│       └── bme_ros2_tutorials_py/
├── build/                     # colcon 產生
├── install/                   # colcon 產生
└── log/                       # colcon 產生
```

`robot_pick_place/` 本身不是 ROS 套件，也不是另一個 workspace。仍在 `ros2_ws/`
根目錄建置；ROS 套件名稱與 `ros2 launch` 指令維持不變。新增其他手臂時，
可把它的專用套件放在 `src/robot_pick_place/` 下，與 `mycobot_ros2/` 並列。

在包含 `src/` 的 workspace 根目錄執行：

目前工作區已備妥官方 Kinova 來源。全新 checkout 請先依
[Kinova 文件的來源還原步驟](src/robot_pick_place/kinova_gen3_lite_support/README.md#官方來源的最小相容修正)
取得固定版本並套用相容 patch；該外部 git checkout 由 `upstream.repos` 管理，不納入主儲存庫。

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

# 原本流程：Gazebo → 相機／點雲感知 → 場景 → MTC；預設只規劃
ros2 launch robot_bringup demo.launch.py \
  robot:=mycobot_280 execute:=false

# 可選的已知物件規劃測試，不啟動相機或感知服務
ros2 launch robot_bringup demo.launch.py \
  robot:=mycobot_280 scene_source:=known execute:=false

# Kinova：相同感知服務與 MTC 程式，外部 RGBD 相機
ros2 launch robot_bringup demo.launch.py \
  robot:=kinova_gen3_lite execute:=false

# Kinova：已知物件規劃測試
ros2 launch robot_bringup demo.launch.py \
  robot:=kinova_gen3_lite scene_source:=known execute:=false
```

無 GUI 時在命令後加 `headless:=true use_rviz:=false`。
確認場景、規劃及控制器後，可用 `execute:=true`；程式會檢查 Controller 與 Action
是否可用。MoveIt 的物件 Attach **不代表** Gazebo 已實現物理夾持。

後續接入機器手臂時，沿用同一套感知與 MTC 任務：

1. 準備該手臂的 URDF/Xacro、SRDF、MoveIt 與控制器設定，以及 Gazebo 啟動檔。
2. 參考 `src/robot_pick_place/robot_bringup/config/robots/mycobot_280.yaml`，新增 `config/robots/<robot_id>.yaml`，填入套件路徑、規劃群組、TCP、夾爪姿態、控制器及相機 topics。
3. 在 `config/scenes/` 新增該場景的任務／感知參數，於 Profile 指向這些檔案。廠商 Launch 參數不同時，增加對應的小型 wrapper，契約見下方架構文件。
4. 建置後使用 `ros2 launch robot_bringup demo.launch.py robot:=<robot_id>`。

正常接入不必修改 `robot_task_core` 的 Pick and Place 步驟，也不必複製感知程式。
不同的夾爪控制方式可能需要 adapter。Kinova Gen3 Lite 2F 沿用同一個 GripperCommand
控制介面與 named_pose 策略，無須新增 MTC 或 Action adapter。

[Kinova 官方來源、相容修正、安裝與測試](src/robot_pick_place/kinova_gen3_lite_support/README.md)

Kinova 場景另含簡化蘋果、橘子、飲料罐與小紙盒；預設仍抓原長圓柱。
可用 `task_params_file` 指向 `robot_bringup/config/scenes/kinova_grasp_can.yaml` 改抓飲料罐，
指令與尺寸見上述 Kinova 文件。新場景的原目標與飲料罐完整規劃均已驗證。

[架構、Profile 契約、舊入口、安裝與排錯](docs/architecture.md) ·
[修改檔案清單](docs/changed_files.md) · [實際驗證結果](docs/validation.md)
