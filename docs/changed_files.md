# 修改檔案清單

初始通用化使用 myCobot 280，後續依要求新增 Kinova Gen3 Lite。使用者提供的規格文件及
Zone.Identifier 未修改；舊 MTC、Service Type、PCL 來源與 robot.sh 保留。

## 本次 Kinova Gen3 Lite 整合

本次未修改 myCobot 套件，也未修改／複製 `robot_task_core` 的任務程式或共用感知演算法。

| 檔案／目錄 | 變更 |
|---|---|
| `src/robot_pick_place/ros2_kortex/` | 官方 Jazzy commit 462dab9；只建置 kortex_description 與 kinova_gen3_lite_moveit_config |
| 官方 `kortex_description/arms/gen3_lite/6dof/urdf/kortex.ros2_control.xacro` | 模擬夾爪主動關節不再與 arm hardware 重複註冊 |
| 官方 `kortex_description/grippers/gen3_lite_2f/urdf/gen3_lite_2f.ros2_control.xacro` | 使用正確的 GazeboSimSystem plugin 名稱 |
| 官方來源內五個 `COLCON_IGNORE` | 排除實體 API／driver／bringup 與兩款非 Lite 的 MoveIt config |
| `src/robot_pick_place/kinova_gen3_lite_support/{CMakeLists.txt,package.xml,LICENSE,README.md,upstream.repos}` | 模擬整合套件、官方來源版本與重建方式 |
| 同套件 `patches/ros2_kortex-jazzy-lite-gazebo.patch` | 記錄兩行 upstream 相容修正，方便重現 |
| 同套件 `urdf/gen3_lite.urdf.xacro` | 呼叫官方模型宏，加入 Harmonic plugin、runtime Controller YAML、可選外部 RGBD 相機 |
| 同套件 `config/{gen3_lite.srdf,initial_positions.yaml}` | 正確末端群組、home/open/closed 姿態與起始關節位置 |
| 同套件 `config/{joint_limits,kinematics,moveit_controllers,ompl_planning,pilz_cartesian_limits}.yaml` | 沿用官方 Lite 設定，KDL timeout 設為 0.1 秒 |
| 同套件 `config/ros2_controllers.yaml` | 六軸軌跡與 2F GripperCommand 控制器，無需另寫 adapter |
| 同套件 `launch/{gazebo,move_group,robot_state_publisher}.launch.py` | 遵循既有共用 bringup 契約，模型生成一次、各核心節點僅一份 |
| 同套件 `worlds/pick_place.sdf`、`rviz/pick_place.rviz` | 可重現的規劃測試場景與 RViz 預設 |
| `src/robot_pick_place/robot_bringup/config/robots/kinova_gen3_lite.yaml` | Kinova Profile，與 myCobot 共用相同載入流程 |
| `src/robot_pick_place/robot_bringup/config/scenes/kinova_{known,perception,perception_settings}.yaml` | 已知場景、放置位置及相機感知裁切參數 |
| `src/robot_pick_place/robot_bringup/package.xml` | 增加 Kinova 整合套件的執行依賴 |
| `tests/run_demo_check.py` | 新增 --robot 選項，節點探索等待 3 秒，保留原隔離／清理方式 |
| `tests/test_kinova_profile.py` | 六軸模型、硬體關節唯一性、mimic／姿態／Controller／sensor 轉換契約 |
| `tests/check_kinova_controllers.py` | 真實模擬 Action 目標、夾爪與手臂關節回饋檢查 |
| `README.md`、`docs/{architecture,changed_files,validation}.md` | 兩種手臂入口、實際測試結果及限制 |
| `.gitignore` | 官方外部 checkout 不作為未設定的 embedded git repository 加入主儲存庫；以 upstream.repos 與 patch 還原 |

後續 `execute=true` 必要修正：Kinova 的 `config/gen3_lite.srdf` 增加 arm_with_gripper；
`config/ros2_controllers.yaml` 增加 allow_nonzero_velocity_at_trajectory_end=true。
`tests/run_demo_check.py` 新增 --execute，必須看到 Task executed successfully 才算通過；
`tests/test_kinova_profile.py` 增加合併軌跡群組的回歸檢查，目前共 16 項 Profile 測試。

後續生活物品場景：`kinova_gen3_lite_support/worlds/pick_place.sdf` 增加簡化蘋果、橘子、
飲料罐、小紙盒，均有動態碰撞與慣性；`robot_bringup/config/scenes/kinova_grasp_can.yaml`
提供可選飲料罐目標。`tests/run_demo_check.py` 新增 --task-params-file 以重現該目標的驗證。
只使用原本的幾何感知與 MTC，未加入水果語意辨識或更動其他機器人場景。
再依要求將四個生活物品移至距底座約 30～41cm，保留原長圓柱；同步更新飲料罐 YAML 位姿及
Kinova README。實際新位置與規劃檢查結果見 validation.md。

抓罐逾時後續診斷：只在 Kinova 的 `config/moveit_controllers.yaml` 增加夾爪
allowed_goal_duration_margin=3.0，留給接觸停滯的完成判定。未停用逾時檢查或修改 MTC。
GUI 模式另重現已安裝 MoveIt 的執行期崩潰，尚未修正，詳細限制記錄於 validation.md。

後續控制器 readiness 卡住修正：`robot_bringup/robot_bringup/readiness.py` 增加單筆
ListControllers 回覆 3 秒逾時、取消 pending request 及重試，仍檢查全部 active。
新增 `tests/check_readiness_retry.py` 的真實 ROS 延遲回覆／inactive 回歸測試；
`tests/run_demo_check.py` 增加 --with-gui，覆蓋使用者的 Gazebo GUI + RViz + execute=true 啟動方式。

## 新套件

| 檔案（均相對 workspace 根目錄） | 責任與原因 |
|---|---|
| `src/robot_pick_place/robot_task_interfaces/srv/GetPlanningScene.srv` | 保留舊欄位語意的通用 ROSIDL 型別，避免新模組依賴 mycobot_interfaces |
| `src/robot_pick_place/robot_task_interfaces/{CMakeLists.txt,package.xml}` | ROSIDL 生成與 runtime export |
| `src/robot_pick_place/robot_task_core/src/mtc_node.cpp` | 唯一通用 MTC Stage 實作、兩種 scene source、RobotModel/參數/Controller 驗證；保留原 Stage 順序，修正 mimic 接觸與 Place IK |
| `src/robot_pick_place/robot_task_core/src/get_planning_scene_client.cpp` | 通用 Service Client，名稱與等待/回覆逾時可設定 |
| `src/robot_pick_place/robot_task_core/include/robot_task_core/get_planning_scene_client.h` | 通用型別與 Client 介面 |
| `src/robot_pick_place/robot_task_core/config/mtc_defaults.yaml` | 共用運動設定；execute=false，機器人專屬數值由 Profile 提供 |
| `src/robot_pick_place/robot_task_core/launch/pick_place.launch.py` | 獨立任務入口，合併 ROS YAML 與明確 Launch 覆寫 |
| `src/robot_pick_place/robot_task_core/{CMakeLists.txt,package.xml}` | 僅 ROS/MoveIt/MTC/通用介面的建置依賴 |
| `src/robot_pick_place/robot_perception/src/get_planning_scene_server.cpp` | 舊 Server 改用通用型別、可設定服務名稱及通用節點名稱 |
| `src/robot_pick_place/robot_perception/src/{cluster_extraction,object_segmentation,plane_segmentation,normals_curvature_and_rsd_estimation}.cpp` | 複用舊 PCL helpers，演算法內容不變，僅調整 include 路徑 |
| `src/robot_pick_place/robot_perception/include/robot_perception/{cluster_extraction,object_segmentation,plane_segmentation,normals_curvature_and_rsd_estimation}.h` | 對應 PCL helpers 的通用套件標頭 |
| `src/robot_pick_place/robot_perception/config/perception_defaults.yaml` | 可覆寫的 topic/frame/service 與感知設定 |
| `src/robot_pick_place/robot_perception/launch/perception.launch.py` | 獨立通用感知入口 |
| `src/robot_pick_place/robot_perception/{CMakeLists.txt,package.xml}` | 獨立編譯 PCL，不依賴 mycobot 套件 |
| `src/robot_pick_place/robot_bringup/config/robots/mycobot_280.yaml` | 目前唯一真實 Robot Profile：資源、Group、Frame、姿態、Controller、感知與場景 |
| `src/robot_pick_place/robot_bringup/config/scenes/mycobot_known.yaml` | 與 world 核對的圓柱中心/碰撞尺寸、支撐面及原放置目標 |
| `src/robot_pick_place/robot_bringup/config/scenes/mycobot_perception.yaml` | 保留舊辨識目標尺寸與放置目標 |
| `src/robot_pick_place/robot_bringup/config/scenes/mycobot_perception_settings.yaml` | 保留原辨識/裁切調校，作為整合層設定 |
| `src/robot_pick_place/robot_bringup/robot_bringup/profile.py` | Profile schema、套件/資源、SRDF 與 Controller 對應驗證 |
| `src/robot_pick_place/robot_bringup/robot_bringup/configuration.py` | 依 Profile 產生 runtime YAML/URDF 與共用 MoveIt 參數，不寫回 source/install |
| `src/robot_pick_place/robot_bringup/robot_bringup/readiness.py` | 等待 active controllers、Joint States、TF、clock、場景服務；感知模式再等資料 |
| `src/robot_pick_place/robot_bringup/launch/demo.launch.py` | 整合既有廠商 Launch，readiness 成功才啟動共用 MTC |
| `src/robot_pick_place/robot_bringup/{package.xml,setup.py,setup.cfg,resource/robot_bringup,robot_bringup/__init__.py}` | ament_python 包裝與 readiness 執行入口 |
| 四個新套件各自的 `LICENSE` | 保留原 BSD 授權 |

## 既有套件的必要整合修改

以下路徑位於 `src/robot_pick_place/mycobot_ros2/`：

| 檔案 | 修改 |
|---|---|
| `mycobot_description/launch/robot_state_publisher.launch.py` | 用 Package Share 範本與獨立暫存 YAML，依序完成產檔再處理 Xacro；可接收整合層已產生的 URDF/YAML |
| `mycobot_description/urdf/robots/mycobot_280.urdf.xacro` | 新增 controllers_file 傳遞；未修改機構、關節與幾何 |
| `mycobot_description/urdf/control/gazebo_sim_ros2_control.urdf.xacro` | plugin parameters 確實使用傳入的 runtime YAML |
| `mycobot_description/package.xml` | 明列使用到的 Launch/Xacro 資源查找依賴 |
| `mycobot_gazebo/launch/mycobot.gazebo.launch.py` | 傳遞 robot_name、runtime URDF/YAML；支援 headless；相機關閉時不啟動 image_bridge |
| `mycobot_moveit_config/launch/move_group.launch.py` | 接收完整共用 MoveIt 字典；未提供時仍走原本 Builder 路徑 |
| `mycobot_moveit_config/launch/load_ros2_controllers.launch.py` | spawner 依 Controller Manager 就緒啟用原控制器，移除固定啟動延遲，失敗明確停止 |
| `mycobot_moveit_config/package.xml` | 補齊新整合路徑的執行依賴 |
| `mycobot_moveit_config/config/mycobot_280/mycobot_280.srdf` | 補上固定相機與支架的 Adjacent 碰撞排除，避免啟用相機後所有抓取解被拒絕 |

## 驗證與說明

| 檔案 | 用途 |
|---|---|
| `tests/test_profiles.py` | 12 項 Profile/資源/異常設定、任意 install prefix、Stage/PCL 保留檢查 |
| `tests/check_perception_interface.py` | 真實通用 Client/Server round trip，無感測資料時應失敗 |
| `tests/check_task_safety.py` | 9 項真實 MTC executable 的異常參數、known 無感知等待、感知 timeout/failure |
| `tests/run_demo_check.py` | 有時限的 execute=false 整合規劃測試，檢查 CLI 與重複節點，只清理自己建立的程序 |
| `README.md` | 實際建置、啟動與說明入口 |
| `docs/architecture.md` | 新舊比較、依賴圖、Profile/後端契約、舊入口、排錯與擴充步驟 |
| `docs/validation.md` | 實際測試結果、未測項目與最後一次 Place IK 證據 |
