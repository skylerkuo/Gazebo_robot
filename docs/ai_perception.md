# Kinova：切換幾何匹配與 YOLOE 辨識

兩種模式共用原本的 `robot_task_core`，不需改 Robot Profile 或新增 MTC 任務。
`scene_source:=perception` 表示從相機建立場景；`perception_method` 決定如何選出抓取目標。

| 模式 | 如何選目標 | 3D 場景來源 |
|---|---|---|
| `geometry`（預設） | 原本的形狀與尺寸匹配 | 原本的 PCL 點雲處理 |
| `ai` | YOLOE 文字類別 → 分割遮罩 → 對齊深度 → 對應碰撞物件 | 同一個 PCL 點雲處理 |

AI 沿用 `object_detect.py` 的 YOLOE、每類保留最高信心實例的方式。整合後改讀 ROS 相機，
而非固定圖片；保留其他物件及支撐面，只有目標 ID 由 AI 決定。
沒有辨識結果、遮罩沒有足夠深度、與 3D 物件無法可靠對應或對應有歧義時，回傳失敗並停止規劃。
不會自動改抓原本幾何匹配的物件。

## 環境準備

幾何模式不需要 Ultralytics 或 PyTorch。AI 使用獨立 `.venv`，並透過 system-site-packages
沿用 ROS Python 套件。以下在 `/home/skylerkuo/ros2_ws` 執行：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select \
  robot_task_interfaces robot_perception robot_task_core robot_bringup
source install/setup.bash

# 僅第一次需要建立並安裝；系統須已安裝 python3-venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r src/robot_pick_place/robot_perception/requirements-ai.txt
```

此工作區已建立 `.venv`。每次開新終端只需 source ROS、workspace 與 `.venv/bin/activate`。
安裝限制 NumPy `<2` 以相容 Jazzy 的 cv_bridge。上例為 CPU；使用 CUDA 需另外安裝與驅動相容的
PyTorch，再設定 `ai_device:=0`。

YOLOE 第一次使用會下載模型、文字編碼器及相關依賴，應先在有網路的環境預熱：

```bash
mkdir -p ~/.cache/robot_perception/yoloe
cd ~/.cache/robot_perception/yoloe
python - <<'PY'
from robot_perception.yoloe_detector import YOLOEDetector
YOLOEDetector('yoloe-26n-seg.pt', ['drink can'], device='cpu')
PY
cd /home/skylerkuo/ros2_ws
```

模型及文字編碼器放在 `ai_cache_dir`（預設上述目錄）。`ai_model` 可指定現有權重的絕對路徑。
YOLOE 的模型、文字提示及下載行為參考 [Ultralytics 官方文件](https://docs.ultralytics.com/models/yoloe/)。

## 啟動

### 終端輸入、多次抓取（建議入口）

source ROS、workspace 與 `.venv/bin/activate` 後，使用一個終端執行：

```bash
ros2 run robot_bringup pick_console --robot kinova_gen3_lite --execute
```

此入口會啟動原 Gazebo、MoveIt、AI 感知和同一份 MTC。就緒後輸入一個英文物品名稱：

```text
要夾取的物品 > drink can
開始處理：drink can；擷取新 RGB／深度 → 辨識 → 規劃 → 執行
完成：Pick and place executed successfully
等待下一個物品。
要夾取的物品 > green plastic bottle
...
要夾取的物品 > quit
```

這是操作介面的格式示例；各物品的實際辨識／規劃／執行結果以 [測試紀錄](validation.md) 為準。
省略 `--execute` 僅規劃；`--headless --no-rviz` 可用於無視窗測試，`--device 0` 可指定已備妥的 CUDA。
終端顯示任務結果及詳細日誌路徑，`quit`／`exit` 或 Ctrl+C 關閉此次啟動的模擬。

每次 Enter 只觸發一次任務，完成或失敗後再等下一個輸入。沒有輸入時，不執行 YOLOE、PCL、
場景更新或 MTC 規劃。相機本身可持續發布原始資料；感知節點僅在服務請求後短暫訂閱，
要求 RGB／點雲時間戳均晚於該請求，取得同步資料後取消訂閱。下一次輸入會重新取像、
更換文字提示並重建場景，不重用上次影像或解答。
RViz 的感知碰撞物件會在收到指令、成功建立場景後出現；Gazebo 物件仍照原世界顯示。

同時送來的第二筆任務直接回報 busy，不排隊執行。找不到目標時不會自動改抓其他東西，
也不會關閉整個模擬；可再輸入其他物品。若執行失敗後 MoveIt 仍有 attached object，
需先恢復／釋放物件，避免下次建圖直接清除仍在夾爪上的物件。

Kinova 預設改為收集箱投放：箱子外尺寸 32×22×10cm，底面中心 `[0.25,-0.52,0]`。
物品底面移到 `[0.25,-0.52,0.18]`，即高於箱緣 8cm 的位置後開夾，交由重力落入箱內；
省略原本向下接近放置表面的階段。不同高度物品仍會自動換算物品中心高度。
箱底與四片箱壁同時建入 MoveIt，保留碰撞檢查；投放後清掉懸空的物件估計位置。
箱子位於目前感知抓取範圍外，已收集的物品不會再作為待抓物件建圖。

設定位於 `robot_bringup/config/scenes/kinova_{known,perception}.yaml`：
`drop_into_bin`、`bin_pose`、`bin_dimensions`、`bin_wall_thickness` 與 `place_pose`。
箱體實際幾何位於 `kinova_gen3_lite_support/worlds/pick_place.sdf`，移動／改尺寸時須同步。
箱子裝滿後需清空；此功能不會排列物品或規劃物品堆疊。
myCobot 保留原放置方式；`drop_into_bin` 預設為 false。

### 使用 launch 與服務

直接使用 AI launch 時，預設改為等待指令，不會自動抓取：

```bash
ros2 launch robot_bringup demo.launch.py robot:=kinova_gen3_lite \
  perception_method:=ai execute:=true
```

這種啟動方式沒有 stdin 轉接，可在另一個終端直接送一筆文字服務請求：

```bash
ros2 service call /pick_place/run robot_task_interfaces/srv/RunPickPlace \
  "{target_label: 'drink can'}"
```

`RunPickPlace` 回應 `success`、`executed` 與 `message`。`executed=true` 只在整套執行成功後回傳。
`interactive:=false` 可保留前一版啟動後自動執行一次的 AI 流程。

### 原幾何模式與單次 AI

原本尺寸匹配，抓飲料罐：

```bash
ros2 launch robot_bringup demo.launch.py robot:=kinova_gen3_lite \
  scene_source:=perception perception_method:=geometry \
  task_params_file:=/home/skylerkuo/ros2_ws/src/robot_pick_place/robot_bringup/config/scenes/kinova_grasp_can.yaml \
  execute:=false
```

AI 辨識飲料罐：

```bash
ros2 launch robot_bringup demo.launch.py robot:=kinova_gen3_lite \
  scene_source:=perception perception_method:=ai \
  interactive:=false ai_target_class:="drink can" ai_device:=cpu execute:=false
```

`ai_target_class` 是單次模式的預設文字提示；互動模式則使用每次終端輸入，不需重啟節點。
目前場景建議輸入 `drink can`、`green plastic bottle`、`tennis ball`。
`bottle` 單獨使用仍容易選到藍色飲料罐；`green plastic bottle` 對新綠瓶更明確。
AI 模式不需以 `task_params_file` 指定目標尺寸，MTC 會使用選中碰撞物件的實際估計尺寸與位姿。
抓取角度仍由 Kinova Profile 的 `grasp_frame_transform` 決定；此整合不更動角度。
確認規劃後，可將 `execute:=false` 改成 `execute:=true`，交由現有控制器在 Gazebo 執行。

辨識影像發布於 `/perception/ai/detections`，可在 RViz 加入 Image 顯示。成功日誌含
`[AI] Selected class=... object=... depth_overlap=...`；失敗會顯示具體原因。

## 適用範圍

AI 選目標仍需要現有 PCL 能重建該物件，支援其 BOX／CYLINDER 碰撞近似。
RGB 與點雲須為相同解析度、像素對齊的 organized PointCloud2，時間差不超過 0.15 秒，
且有點雲座標到 `target_frame` 的 TF。現在 Kinova 的 RGBD 相機提供這些資料。
此處沒有新增任意物件的 6D 姿態估計、球體專用抓取或自動抓取角度生成。
Gazebo 的簡化水果、純色幾何模型也可能辨識失敗，需以實際測試結果判定。
目前已移除蘋果與紅色長圓柱，新增綠色瓶子與有白色接縫的網球。
紙盒已移除，換成帶把手與深色杯口的紅色馬克杯；建議輸入 `red cup`，亦可用 `red mug`。
杯身以圓柱近似，視覺把手不參與物理接觸；既有 BOX 位姿估計偏移未在此次修改中修正。

`scene_source:=known` 仍為原本的已知物件模式，不能與 `perception_method:=ai` 合用。
既有 MoveIt 執行／關閉崩潰的限制另記於 [validation.md](validation.md)，不因辨識模式切換而消失。

原圖片程式仍可獨立使用：

```bash
python object_detect.py --source gazebo_scene.png --classes cube --device cpu --no-show
```
