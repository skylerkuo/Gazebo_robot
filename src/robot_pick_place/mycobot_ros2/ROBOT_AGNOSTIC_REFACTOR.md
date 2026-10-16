# Gazebo_robot：建立可重用的多機器人任務架構（Codex 實作規格）

> **給 Codex 的指令：請直接在此儲存庫實作本文件，不要只提出規劃。**
>
> 專案：<https://github.com/skylerkuo/Gazebo_robot>
>
> 目前驗證機器人：**myCobot 280**。本次**不要**加入 Kinova、UR、xArm、Franka 的模型或驅動。
>
> 開發環境：Ubuntu 24.04 / ROS 2 Jazzy / Gazebo Harmonic / MoveIt 2 / MoveIt Task Constructor（MTC）/ colcon，可能在 WSL2 執行。
>
> 文件的目的：把「**同一套任務讓不同機器手臂執行**」變成專案本身的架構能力；不是只優化 myCobot 程式品質，也不是現在去移植第二隻手臂。

---

## 0. 任務定義與成功標準

### 最重要的成果

1. **同一個通用 MTC Pick-and-Place 節點**定義並執行抓取、搬運、放置任務，不應引用 `mycobot_*` 專屬標頭、固定 Group 名稱、關節名稱、末端 Frame 或控制器名稱。
2. **機器人專屬設定**（URDF/Xacro、SRDF、IK、Joint Limits、MoveIt Controllers、ros2_control、夾爪、TF 等）留在對應機器人的 Package；通用任務透過「Robot Profile + Launch 整合層」取得所需設定。
3. **感知與抓放解耦**：能以已知座標（`known`）建立場景，不啟動相機也能規劃；也能選擇現有點雲流程（`perception`）建立同一種 MoveIt Planning Scene。
4. **保留既有 myCobot 完整能力**：Gazebo、MoveIt、Controller、相機點雲、支撐面/物件辨識、MTC 全套 Stage；原本入口盡可能保持可執行，提供相容/遷移方式。
5. **未來加機器人不用複製任務與感知程式**：只需要新增 Profile，引用新廠商的 Description/MoveIt/Gazebo 套件，並在必要時提供小型 Controller/Gripper Adapter。無法完全通用的地方要有明確介面和說明。

### 本次不做的事

- 不安裝、不引入、不實作第二種機器手臂。
- 不修改或重寫官方機器人運動學、URDF Mesh、SDF 模型、PCL 分割演算法，除非編譯、介面移轉或修正明確錯誤所必需。
- 不建立大型 Plugin Framework、抽象工廠、複雜繼承樹、資料庫、微服務或 AI Agent。
- 不要求在 Gazebo 中立即完成物理夾持：MTC `attachObject()` 修改的是 **MoveIt Planning Scene**，不等於 Gazebo 物理模型自動固定在夾爪上。不得把 MoveIt 物件附著視為 Gazebo 真正抓取的證據。
- 不破壞現有 myCobot 啟動方式；不能做到完全相容時，提供等效包裝入口並明列差異。

### 安全原則

- 修改前先 `git status --short`、記錄當前 Commit/Branch，請勿覆蓋使用者未提交的變更。
- 在功能正常前，**保留既有 `src/mycobot_ros2/` 來源**作為對照；新通用 Package 逐步導入。禁止一開始整包搬走、刪除或大量重新命名。
- 每完成一個 Phase 執行相應檢查；失敗時修正，再繼續下個 Phase。
- 不要用 `pkill -9 -f 'ros2|gz|...'` 這類廣泛清理方式；新啟動器應正常管理自己的程序。
- 預設 `execute:=false`（只規劃、不下發運動），在確認規劃、碰撞、座標與控制器配置正確後才測試執行。

---

## 1. 已核對的現有程式（以本文件編寫時 main 分支為準）

儲存庫已有多個 ROS 2 Package；實作時先重新讀取本地最新版，以實際程式為準。

```text
src/mycobot_ros2/
├── mycobot_description/
│   ├── urdf/robots/mycobot_280.urdf.xacro
│   ├── urdf/control/
│   ├── urdf/sensors/intel_rgbd_cam_d435.urdf.xacro
│   └── launch/robot_state_publisher.launch.py
├── mycobot_gazebo/
│   ├── launch/mycobot.gazebo.launch.py
│   ├── config/ros_gz_bridge.yaml
│   └── worlds/pick_and_place_demo.world
├── mycobot_moveit_config/
│   ├── config/mycobot_280/{mycobot_280.srdf,kinematics.yaml,moveit_controllers.yaml,ros2_controllers.yaml,...}
│   └── launch/{move_group.launch.py,load_ros2_controllers.launch.py}
├── mycobot_interfaces/srv/GetPlanningScene.srv
└── mycobot_mtc_pick_place_demo/
    ├── src/mtc_node.cpp
    ├── src/get_planning_scene_client.cpp
    ├── src/get_planning_scene_server.cpp
    ├── src/{cluster_extraction,object_segmentation,plane_segmentation,normals_curvature_and_rsd_estimation}.cpp
    ├── include/mycobot_mtc_pick_place_demo/*.h
    ├── config/{mtc_node_params.yaml,get_planning_scene_server.yaml}
    ├── launch/{pick_place_demo.launch.py,get_planning_scene_server.launch.py}
    └── scripts/robot.sh
```

### 發現的耦合點（請逐項處理）

**A. MTC 節點** `mycobot_mtc_pick_place_demo/src/mtc_node.cpp`

- 已將 `arm_group_name`、`gripper_group_name`、`gripper_frame`、開合姿態、`world_frame`、`controller_names`、`grasp_frame_transform` 等參數化：**保留此優點**。
- 但預設值仍是 myCobot 的 `arm`、`gripper`、`link6_flange`、`base_link`、`arm_controller`、`gripper_action_controller`。
- `main()` 目前先固定 `setupPlanningScene()`（呼叫自訂 Service），再 `doTask()`；無相機/Service 時不能進入任務。
- Grasp/Place Stage（`GenerateGraspPose`、`ComputeIK`、`MoveTo`、`MoveRelative`、`ModifyPlanningScene` 等）本身可以作為通用邏輯，盡量原樣保留。
- 誤用 SRDF group、IK link、命名姿態、Planner config 或 controller 時，應有明確失敗訊息。

**B. 感知 Service**

- Client：`src/get_planning_scene_client.cpp` 目前硬編碼 `/get_planning_scene_mycobot`，會持續等待 Service。
- Server：`src/get_planning_scene_server.cpp` 用 `mycobot_interfaces/srv/GetPlanningScene`、固定 Service 名 `get_planning_scene_mycobot`，接收 PointCloud2，產生 `PlanningSceneWorld`。
- Service 定義位於 `mycobot_interfaces/srv/GetPlanningScene.srv`。
- 現有 Server 的分割支援等保留，不要為了更名而重寫核心 PCL 演算法。

**C. 啟動與資源查找**

- `mycobot_gazebo/launch/mycobot.gazebo.launch.py` 固定引用 `mycobot_description`、`mycobot_moveit_config`。
- `mycobot_mtc_pick_place_demo/launch/pick_place_demo.launch.py` 的 `MoveItConfigsBuilder` 固定 `mycobot_moveit_config`，即使帶不同 `robot_name` 也不等於能換機器人。
- `mycobot_description/launch/robot_state_publisher.launch.py` 中 `process_ros2_controllers_config()` 以 `~/ros2_ws/src/mycobot_ros2/...` 等路徑讀寫 YAML，並可能寫入 source/install 目錄；需要改成不綁 Workspace 名稱、避免不必要副作用。
- 舊 `scripts/robot.sh` 有多個 `sleep` 與廣泛 `pkill`，**新流程**要可靠而保守，舊腳本先留著。

**D. 感知座標與夾爪**

- 現有 Camera 設在 myCobot 相關 URDF 下，topic 包含 `/camera_head/depth/color/points`。
- `get_planning_scene_server.yaml` 的 `target_frame: base_link`、裁切及支撐面參數和目前場景耦合；未來應隨 Profile/Scene 設定，不能由通用任務寫死。
- myCobot 的 MoveIt 夾爪控制為 `GripperCommand`。其他品牌可能不是同一 Action/控制模式：設計**可插入轉接介面**，但這次只實作 myCobot 的可用路徑，不要憑空實作其他品牌。

---

## 2. 目標架構（要實作，不只是文件示意）

保留既有供應商/機器人 Package；在專案 `src/` **新增以下四個 ament Package**。若本地已有同名 Package，先檢查與沿用，不製造重名。

```text
Gazebo_robot/
├── ROBOT_AGNOSTIC_REFACTOR.md    # 本開發規格
├── README.md                     # Codex 補上實際使用說明（若原本不存在則新增）
└── src/
    ├── robot_task_interfaces/     # 通用的 GetPlanningScene.srv
    │   ├── srv/GetPlanningScene.srv
    │   ├── CMakeLists.txt
    │   └── package.xml
    │
    ├── robot_task_core/           # 唯一通用的 MTC Pick and Place 實作
    │   ├── src/mtc_node.cpp
    │   ├── src/get_planning_scene_client.cpp
    │   ├── include/robot_task_core/get_planning_scene_client.h
    │   ├── config/mtc_defaults.yaml
    │   ├── launch/pick_place.launch.py
    │   ├── CMakeLists.txt
    │   └── package.xml
    │
    ├── robot_perception/          # 與品牌無關的感知與場景重建
    │   ├── src/{get_planning_scene_server,cluster_extraction,object_segmentation,
    │   │        plane_segmentation,normals_curvature_and_rsd_estimation}.cpp
    │   ├── include/robot_perception/*.h
    │   ├── config/perception_defaults.yaml
    │   ├── launch/perception.launch.py
    │   ├── CMakeLists.txt
    │   └── package.xml
    │
    ├── robot_bringup/             # 載入 Robot Profile、統一啟動
    │   ├── config/robots/mycobot_280.yaml
    │   ├── launch/demo.launch.py
    │   ├── robot_bringup/         # 如使用 ament_python，放 profile 解析/驗證工具
    │   ├── package.xml
    │   └── [setup.py 或 CMakeLists.txt，依實作選擇]
    │
    └── mycobot_ros2/             # 保留既有機器人專屬程式與舊 Demo
        ├── mycobot_description/
        ├── mycobot_gazebo/
        ├── mycobot_moveit_config/
        ├── mycobot_interfaces/  # 舊相容介面，不作為新通用模組依賴
        ├── mycobot_mtc_pick_place_demo/  # 舊 Demo 保留直到相容驗證完成
        └── ...
```

### 職責與依賴規則（強制）

```text
robot_bringup (選擇 robot Profile 並啟動)
    ├── 選用 mycobot_* (目前唯一後端；未來新增新廠商套件)
    ├── 啟動/設定 MoveIt + Gazebo + ros2_control
    ├── 啟動 robot_perception（僅 perception 模式）
    └── 啟動 robot_task_core
                         │
                         ▼
                robot_task_core
             ├── MoveIt 2 / MTC
             ├── robot_task_interfaces
             └── ROS 2 參數

 robot_perception ───> robot_task_interfaces
```

- `robot_task_core` **不得依賴** `mycobot_description`、`mycobot_moveit_config`、`mycobot_interfaces`、`mycobot_gazebo` 或廠商 SDK。
- `robot_perception` **不得依賴** `mycobot_*` 套件或特定機器人 URDF。
- 廠商專屬的 URDF/ros2_control/kinematics/SRDF/夾爪控制與相機安裝設定，保留在機器人相應 Package 或 `robot_bringup` Profile，不複製到 `robot_task_core`。
- **不要將官方/既有 URDF/MoveIt Package 的所有內容搬進 `vendor_robots`**。ROS 2 Package 放在目前 `src/mycobot_ros2/` 已足夠；抽象化的重點是**依賴方向**，不是深層資料夾名稱。
- 不能只把 `mtc_node.cpp` 複製到新 Package 就算完工；`package.xml`、`CMakeLists.txt`、Launch、Service Type、Parameter 及執行流程都要真正對齊。

---

## 3. Robot Profile 契約

先只實作 `robot_bringup/config/robots/mycobot_280.yaml`，**不要**建立尚未可運作的 `kinova.yaml`/`ur5e.yaml` 假配置。

Profile 必須描述足夠資訊，讓 bringup 選擇對應廠商套件，而 MTC 取得任務所需參數。可依 ROS 2 Launch 特性調整欄位，但請維持一致、可驗證的格式，並寫明 schema。

建議形狀（欄位與值需由 Codex 對照**實際套件**再確認，不能只照抄）：

```yaml
robot_id: mycobot_280

simulation:
  package: mycobot_gazebo
  launch_file: mycobot.gazebo.launch.py
  world_file: pick_and_place_demo.world

moveit:
  package: mycobot_moveit_config
  launch_file: move_group.launch.py
  robot_name: mycobot_280

robot_description:
  package: mycobot_description
  launch_file: robot_state_publisher.launch.py

planning:
  arm_group: arm
  gripper_group: gripper
  end_effector: gripper
  ik_frame: link6_flange
  base_frame: base_link
  home_pose: home
  open_pose: open
  close_pose: half_closed
  grasp_frame_transform: [0.0, 0.0, 0.096, 1.5708, 0.0, 0.0]

execution:
  arm_controller: arm_controller
  gripper_controller: gripper_action_controller

perception:
  point_cloud_topic: /camera_head/depth/color/points
  rgb_image_topic: /camera_head/color/image_raw
  target_frame: base_link
```

**必要要求：**

1. 由 `robot_bringup` 的 Profile 解析器讀取並驗證欄位，轉成 Launch 引數及對 MTC/Perception 節點的 ROS Parameter；不可叫 C++ MTC 直接讀取這份帶有機器人啟動資訊的自訂 YAML。
2. 不做字串搜尋取代程式碼、也不假設各廠商 Config 檔案都具有 `config/<robot_name>/...` 相同路徑。**以 Package 與 Launch 入口做整合**，不同套件由個別後端整合策略解決。
3. 未來替換機器人時，Profile 可指出其 Description/MoveIt/Controller Launch，**不需重新編譯或修改 MTC Stage 邏輯**。若不同夾爪的 Action 型別不相容，需另做 Adapter，不能假裝只改 `gripper_controller` 字串就會工作。
4. `end_effector`、`gripper_group`、`ik_frame` 在 MoveIt 中的意義不同；依 SRDF 真實定義核對；不要假設它們是同一個字串。
5. 禁止 `~`、使用者名稱、固定絕對路徑或 `~/ros2_ws` 出現在通用程式的資源定位邏輯；ROS 套件路徑應使用 `ament_index_python`、`FindPackageShare` 等機制。
6. 必須清楚檢查 Config 不存在、Package 不存在、SRDF Group 不存在、Frame 不存在、控制器不一致等錯誤，並在啟動時報出具體欄位與預期名稱。

---

## 4. 感知的通用介面與 Scene Source

### 4.1 建立通用 Service Type

從現有 `mycobot_interfaces/srv/GetPlanningScene.srv` 保留資料欄位/語意，建立：

```text
robot_task_interfaces/srv/GetPlanningScene.srv
```

- 使用 ROSIDL 正確設定 `CMakeLists.txt`、`package.xml`（必要的依賴：`moveit_msgs`、`sensor_msgs`、`rosidl_default_generators` / `rosidl_default_runtime`）。
- 修改新 `robot_perception` Server 與新 `robot_task_core` Client 使用 `robot_task_interfaces/srv/GetPlanningScene`。
- Service 名稱改成由參數控制，預設例如 `/get_planning_scene`（務必 Client/Server 一致）。
- 舊版 `mycobot_interfaces` 與 `/get_planning_scene_mycobot` 保留作為舊 Demo 的相容路徑；不可在新舊 Service 間混用不同 ROSIDL Type。
- 如果舊 Demo 和新模組同時啟動，**避免 Service/Node 名衝突**；文件列出互斥啟動注意事項。

### 4.2 把 perception 搬離 MTC

- 將 `get_planning_scene_server.cpp` 與 PCL helpers / 對應 Headers/CMake 依賴搬成 `robot_perception` 的獨立編譯目標。**先複製及調通，再整理舊程式，不要直接刪除。**
- 保留可辨識 box/cylinder、支撐面、點雲座標轉換、裁切、Shape Fitting 與原有 log 功能。
- Topic、Frame、相機參數、平面高度容差、crop 範圍等讀自參數檔，不在程式裡綁 myCobot 命名。
- 感知失敗（無點雲、TF 失敗、找不到支撐面/目標、Service 逾時）應回傳或報出**明確錯誤**，不能在無有效場景時繼續執行。

### 4.3 MTC 要支援兩種 Scene Source

- `scene_source=known`：不啟動相機和感知 Service；從 ROS Parameters 建立目標物件及支撐面 CollisionObject，透過 MoveIt Planning Scene Interface 套用；保留目標 ID、支撐面 ID，以便現有 `allowCollision`、`attachObject`/`detachObject` 等 Stage 使用。
- `scene_source=perception`：透過通用 `/get_planning_scene` Service 取得 `PlanningSceneWorld`、`target_object_id`、`support_surface_id`，套用後照原本 Stage 流程執行。
- `known`/`perception` 必須最後進入**同一個 `createTask()`**（或同等的共用 TaskBuilder），不能各寫一套完整 MTC Stage。
- Task 的場景存取/辨識來源與 MTC Stage 分離成簡潔函式或小型 class；不要求為了兩種來源導入大型策略框架。
- 物件 Pose、Frame、四元數、Shape dimensions 要依 ROS 定義處理：`CYLINDER = [height, radius]`、`BOX = [x, y, z]`；不要把 BOX 的 X 長度誤認為物件高度。擺放高度調整須考慮物件在放置方向的實際尺寸。
- 保留 `get_planning_scene_client` 等舊實作供舊入口驗證，但新 Node 不可依賴 `mycobot_interfaces`。
- Service Client 要有可設定逾時及錯誤處理；避免永遠無限等待。

---

## 5. 共用 MTC Task 規格

### 5.1 保留的任務語意

從原 `mtc_node.cpp` 重用下列流程與使用到的 MoveIt Task Constructor Stage：

```text
CurrentState
  → OpenGripper
  → Connect/MoveToPick
  → Approach
  → GenerateGraspPose + ComputeIK
  → AllowGripperObjectCollision
  → CloseGripper
  → AllowObjectSupportCollision
  → AttachObject
  → Lift
  → DisallowObjectSupportCollision
  → Connect/MoveToPlace
  → Lower + GeneratePlacePose + ComputeIK
  → OpenGripper
  → DisallowGripperObjectCollision
  → DetachObject
  → Retreat
  → MoveHome
```

- 不改變 Stage 的主要順序、Planner 原理、碰撞處理意圖、Pick/Place 行為；僅做可攜性與必要的錯誤修正。
- `robot_task_core` 應只知道 ROS/MoveIt 的 group、frame、named pose、controller list、TCP/grasp transform、物件與支撐面；**不得有機器人型號條件分支** `if (robot == "mycobot")` 寫在 TaskBuilder 內。
- `MoveItConfigsBuilder` 不能硬編碼 `mycobot_moveit_config`：由 `robot_bringup` 負責解析正確的 MoveIt Package 並向 task node 傳入與 `move_group` **相同的 `robot_description`、`robot_description_semantic`、kinematics、planning pipelines、joint limits 及 trajectory execution 設定**。
- 避免兩個 `move_group`、兩套 Controller Manager、兩個重複 Robot State Publisher；在 Launch 中檢查既有被 include 的 Launch 會啟動哪些 Node。
- `use_sim_time` 必須一致傳給 MoveIt、MTC、Perception、Robot State Publisher、相關 TF 節點；不得隱藏時鐘不一致造成的 TF jump。
- 若某機器人不支援目前 `MoveTo` 夾爪策略，預留 `GripperCommand`/Action/Adapter 的**介面契約與失敗訊息**，此階段只提供 myCobot 的有效整合，不實作其他廠牌 Adapter。

### 5.2 啟動前驗證

- 驗證 Group、Frame、命名姿態、joint/controller 名稱是否與載入的 URDF/SRDF/MoveIt Config 一致；使用可取得的 MoveIt RobotModel API。
- 檢查 scene source、object primitive、尺寸、Pose、必要參數和 TF 是否合理。
- 不能把規劃失敗當成功；`task.init()`、`task.plan()`、`task.execute()` 錯誤要有可辨識 log 和正確處理。
- 預設 `execute=false`。`execute=true` 只有在 ROS Controller 可用且規劃成功時才執行。

---

## 6. 統一 Launch 與 robot_bringup 的使用介面

### 必須做出可實際使用的入口

```bash
# 只規劃、使用已知物件座標（第一個通過的主要測試）
ros2 launch robot_bringup demo.launch.py \
  robot:=mycobot_280 scene_source:=known execute:=false

# 啟動現有相機/點雲感知後再進行規劃
ros2 launch robot_bringup demo.launch.py \
  robot:=mycobot_280 scene_source:=perception execute:=false

# 已確定安全、可用控制器及場景後，測試運動執行
ros2 launch robot_bringup demo.launch.py \
  robot:=mycobot_280 scene_source:=known execute:=true
```

這三個入口為本次要求的**目標使用者介面**，實作時要做到與文件相同（不能只在 README 列出，但實際 Launch 不接受）。

### Launch 行為要求

1. 根據 `robot` 選取 Profile，任何其他名稱要有明確錯誤，不能默默回退 myCobot。
2. 目前使用既有 `mycobot_gazebo`、`mycobot_description`、`mycobot_moveit_config` 啟動機器人；不複製整份廠商 Launch 進 `robot_bringup`。
3. 設定 `scene_source`、`execute`、`use_sim_time`、Gazebo world、是否開 RViz、是否開相機，並將必要值正確傳入各 Node。
4. `known` 模式不依賴 Perception Server；`perception` 模式必須啟動感知所需的 Sensor/ROS-GZ Bridge、Service，確認可用後才開始建立 Task。
5. 避免用一串固定秒數 `sleep` 猜測服務是否就緒；盡量利用 Launch 事件與明確 readiness 檢查（例如 Controller State、Action/Service availability），對等待設定 timeout 並回報原因。
6. 禁止重複 Gazebo/MoveIt/Controller/TF 節點；GUI/WSL 無法正常開啟時提供可選 `use_rviz:=false`、`headless:=true` 等（若實際支援，請實作並文件化）。
7. Profile 查詢與新 Package 可在不同 Workspace 名稱與任意安裝 prefix 使用；一律透過 ROS 2 套件索引取得路徑。

### 原本 myCobot 控制器 YAML 的路徑問題

請整理 `mycobot_description/launch/robot_state_publisher.launch.py` 中 `process_ros2_controllers_config()`：

- 不能依賴 `~/ros2_ws` 或在啟動時任意寫入原始碼/唯讀 install 目錄。
- 優先用參數化、臨時目錄或受管理的 runtime 輸出，並讓 `gz_ros2_control` 取得同一份正確 Config。
- 避免改動原 ROS 2 Control 控制器的行為；確保 `arm_controller`、`gripper_action_controller`、`joint_state_broadcaster` 與原有關節相容。
- 若修正方案涉及 Gazebo 讀檔時機，實際驗證設定在 Xacro 轉換、Gazebo spawn 和 Controller 初始化時均可用。

---

## 7. 明確的實作順序（請按順序做、每階段驗證）

### Phase 0 — 先建立 Baseline

1. 讀取 Repo 的 `README`、上述指定檔案及各 Package 的 `CMakeLists.txt`、`package.xml`。
2. 記錄現有 Package、啟動流程、Controller/Group/Frame、Service Type/Name、ROS-GZ Topic、待修項目。
3. 如果環境允許，使用現有方式 Build 和啟動 myCobot，記錄可運行項目；環境不允許則明確註明，不准宣稱已測成功。
4. 確認 `git status`，在使用者既有變更上謹慎實作；可分階段 commit，但不得擅自強制覆蓋 main 或清空 Workspace。

### Phase 1 — 建立 generic interface 與 perception Package

1. 新增 `robot_task_interfaces`，建出 `GetPlanningScene.srv`。
2. 新增 `robot_perception`：移入/複用 Server、PCL 實作與 Headers，改用新 Service Type。
3. Topic、Frame、Service 名與 crop/plane 參數化；`colcon build --packages-up-to robot_perception` 可成功。
4. 原 `mycobot_*` Package 仍可 Build，不讓 ROSIDL/CMake 命名衝突。

### Phase 2 — 建立 robot_task_core

1. 抽出原 `mtc_node.cpp` 及 Client 作為通用 Package，改用通用 Service Type。
2. MTC Stage 流程保留；從 ROS Parameter 載入 robot group/frame/controller/grasp 設定。
3. 分出 `setupSceneFromPerception()` 和 `setupSceneFromKnownObjects()`，共用 MTC TaskBuilder。
4. 加入失敗訊息、參數檢查、逾時處理；新的 Package 無任何 `mycobot_*` include 或 link 依賴。
5. Build `robot_task_core` 並測試至少能初始化 RobotModel/Task（可先使用現有 MoveIt 啟動流程）。

### Phase 3 — 加入 Profile 與統一 bringup

1. 新增 `robot_bringup` 與唯一真實 Profile `mycobot_280.yaml`。
2. 實作 `demo.launch.py`，整合原有 myCobot Gazebo/MoveIt 與新的 Task/Perception。
3. 確保 Robot Description、Controller、MoveIt MTC 參數一致；只啟動一份各類核心節點。
4. `known` 模式無相機就可以啟動至 Task Plan；`perception` 模式保留目前圖像/點雲/辨識流程。
5. 修正 myCobot runtime YAML 路徑副作用，先驗證不影響原機器人 Controller。

### Phase 4 — 回歸測試與兼容

1. 比較原流程和新流程：RobotModel、各控制器、TF、MoveIt Group、MTC Stage 順序、碰撞幾何、物件抓取與放置的目標。
2. 驗證舊啟動入口仍可執行，或提供同功能 wrapper 與文件說明；原程式不先刪除。
3. 新 `known` 模式完成 Task 規劃測試；新 `perception` 模式完成 Service → Planning Scene → Task 規劃測試（前提為相機/WSL 環境可用）。
4. 成功後再整理不再使用的重複程式。注意舊 Package 需可 Build，否則明確遷移/保留策略，不能留一份會破壞整個 Workspace Build 的殘缺 Package。

### Phase 5 — 更新說明與交付

在 `README.md` 或 `docs/architecture.md` 明確提供：

- 現有與新架構的差異（Package 職責與依賴圖）
- 從空白 ROS 2 Jazzy workspace 安裝依賴、Build、Source 指令
- `known` 和 `perception` 兩種啟動範例；RViz/GUI 無法啟動時的選項
- 目前支援的機器人：**myCobot 280**（其他機器人尚未整合）
- 未來新增另一隻手臂的 checklist：廠商 packages、SRDF Groups、MoveIt/Controller、夾爪 Action/Adapter、TCP/抓取姿態、Gazebo、相機/TF，以及驗證順序
- 典型錯誤排查（PackageNotFoundError、TF/frame、controllers inactive、service timeout、IK / planning failed）
- 實際執行的測試結果與任何尚未驗證的事項

---

## 8. 必須提供的 Build 與驗收指令

以下使用**包含 `src/` 的 Workspace 根目錄**；請不要假設它一定叫 `~/ros2_ws`。Codex 需確定命令可在本 Repo 的實際布局執行。

```bash
# 在 Workspace 根目錄執行
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

# 驗證 Package 被安裝
ros2 pkg prefix robot_task_interfaces
ros2 pkg prefix robot_task_core
ros2 pkg prefix robot_perception
ros2 pkg prefix robot_bringup

# 驗證 Launch 介面
ros2 launch robot_bringup demo.launch.py --show-args

# 測試場景（只規劃）
ros2 launch robot_bringup demo.launch.py robot:=mycobot_280 scene_source:=known execute:=false
ros2 launch robot_bringup demo.launch.py robot:=mycobot_280 scene_source:=perception execute:=false

# 執行期間在另一個已 source 的終端確認
ros2 control list_controllers
ros2 topic list
ros2 service list
ros2 action list
```

### 驗收清單：每個都要明確給出 Passed / Failed / Not Tested

- [ ] `colcon build --symlink-install` 所有相關 Package 成功。
- [ ] `robot_task_core` 不含任何 `mycobot_*` Package 編譯依賴／硬編碼路徑。
- [ ] `robot_perception` 不含任何 `mycobot_*` Package 編譯依賴。
- [ ] `robot_task_interfaces` 的 Service Type 生成、Client/Server 正確相容。
- [ ] `mycobot_280.yaml` 真正被 bringup 解析並傳遞，不是擺著未使用。
- [ ] `known` 模式不啟動感知 Server 仍能建構 Planning Scene 並進行 MTC 規劃。
- [ ] `perception` 模式能取得點雲、支撐面與目標物件，進入相同 MTC Task。
- [ ] Task 讀取 Profile 取得 arm group、gripper group、IK frame、named poses、Controller 名稱。
- [ ] MoveIt/Task/Controller 使用一致 Robot Description、SRDF、時間來源。
- [ ] Gazebo myCobot 手臂模型可生成，Controller/Joint States/TF 正常。
- [ ] 無意外重複的 move_group / controller_manager / robot_state_publisher。
- [ ] `execute=false` 確實不下發運動。
- [ ] 原本 myCobot 全套流程沒有被無預警破壞。
- [ ] 新增 `README.md` 或 `docs/architecture.md`，包含實際指令與故障排查。
- [ ] 未來要加另一機器人時，**不用複製或修改 `robot_task_core/src/mtc_node.cpp` 的任務 Stage**。未來機器人是否真的可用，要待該機器人整合驗證，不能以本次單機測試冒稱全機型兼容。

### 如無法在 Codex 環境中啟動 GUI / Gazebo

- 仍須完成實際程式修改與可執行的靜態檢查、Package Build（若相關依賴存在）。
- 對無法跑的動態測試，列出 **Not Tested** 與阻礙原因，附上使用者可在 WSL2 上執行的正確指令及期待觀察結果。
- 不得編造測試通過，也不得直接把 ROS 2 安裝失敗當成程式碼修改成功。

---

## 9. Codex 最後回覆格式

完成程式碼修改後，以**繁體中文**摘要並提供：

1. **修改/新增的 Package 與主要檔案**：每個說明責任與原因。
2. **新的依賴與資料流**：說明 Robot Profile → MoveIt/Gazebo → MTC，以及 known/perception 兩條場景來源如何匯合。
3. **如何啟動**：列出 `colcon build`、`source`、兩種 `ros2 launch` 指令。
4. **實際驗證結果**：Build、ROS 2 Launch、MoveIt Plan、Controller、TF、Perception、舊版回歸分別標示 Passed / Failed / Not Tested。
5. **未完成事項**：明確列出阻礙與下一步，不要留沒有引用的 stub 檔案或只寫 TODO 就算完成。
6. **日後如何增加另一機器人**：要改的 Profile/Launch/Adapter 清單，和原 MTC 不需改的證據。

> **再強調一次：當前只實作 myCobot 280 作為第一個後端，但架構必須讓未來同一份 MTC 任務可接其他機器人。請以可運作、可編譯、少量必要抽象、可逐步驗證為優先，不要只是建立資料夾或重新命名程式。**
