# ACT 中型模型訓練（2026-10-21，checkpoint format 2）

新版預設使用中型 DETR ACT，微調 ImageNet ResNet18。沿用原本 .venv、三視角資料集與
「未來實測關節位置」標籤，無需重新收集資料，不需要提供最終姿態或文字。
這次只修改與測試程式，沒有啟動正式資料集訓練。

舊版 format 1 模型不再載入；訓練、續訓與 Gazebo 推理統一使用 format 2。
請建立新訓練回合，不要對之前 95 分鐘的模型執行 --resume。

## 執行指令

已安裝 CUDA PyTorch 的原環境不必重新安裝套件，也不需要 colcon build：

~~~bash
cd ~/ros2_ws
source .venv/bin/activate
export PYTHONPATH="$PWD/src/robot_pick_place/robot_bringup:$PYTHONPATH"

python -m robot_bringup.train_act \
  --dataset /mnt/d/robot_datasets/bottle_pick_place_rgb \
  --output /mnt/d/robot_models/act_bottle_medium \
  --preset medium \
  --epochs 2000 \
  --batch-size 4 \
  --accumulation-steps 2 \
  --max-minutes 0 \
  --workers 2 \
  --device cuda:0
~~~

ResNet18 預設會微調，等同加上 --train-backbone。
--max-minutes 0 表示不限時間；本版不承諾 95 分鐘內完成訓練。
正式訓練的總耗時受 GPU、磁碟與採樣次數影響，仍需依進度量測。

**訓練 epoch 的意義已改變**：訓練預設仿照官方 EpisodicDataset，每個 episode 隨機選一個起點，
並在下次讀取該 episode 時重新抽樣。一個 epoch 不會遍歷整份資料集所有影格。
例如有 85 個訓練 episodes，batch 4／累積 2，每 epoch 約 11 次 optimizer 更新；
2000 epochs 約 22,000 次更新。不能直接與舊版逐影格的 3～4 epochs 比較。
官方 README 的模擬範例也使用 [2000 epochs、lr=1e-5](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/README.md)。

資料掃描時先檢查成功狀態及每個影像路徑，D 槽大量 JPEG 小檔案可能花數分鐘。
訓練每個 epoch 從各回合抽樣，並非再次完整掃描所有 JPEG。

## 增加每輪訓練量，降低驗證占比

加上 `--samples-per-episode 10`，每個訓練 demo 每個 epoch 會抽取 10 個時間點。
每次存取都重新隨機取起點，可重複抽到同一個時間點；不是把已選好的 86 筆資料原封不動重播。
各 demo 的抽樣次數相同，不因示範長短而改變權重。訓練標籤、loss、batch size 與固定 VAL 都沿用原設定。

以 86 個訓練 demo、batch 4／累積 2 為例：

| 每個 demo 每輪取樣數 | 每輪訓練樣本 | 每輪 optimizer 更新 | 每輪結束的 VAL |
|---|---:|---:|---:|
| 1（相容原設定） | 86 | 11 | 固定 1,024 筆 |
| 10 | 860 | 108 | 固定 1,024 筆 |

啟動資訊與 metrics.jsonl 會顯示／保存每輪樣本數、optimizer 更新數。
`--samples-per-episode` 預設為 1；續訓可明確覆寫，省略時沿用 checkpoint 設定。
舊 checkpoint 沒有這個欄位時視為 1。此參數只適用於 `--sampling episode`。
如果明確開啟 `--validate-every-steps`，仍可能在 epoch 中途額外驗證。

**增加每輪訓練量後，總 epoch 數也應按預期更新量重新安排。**
例如已完成 501 個原規模 epoch、累積 5,511 次更新，原本訓練到 2,000 輪共約 22,000 次更新。
改成每個 demo 取 10 次後，訓練到第 654 輪約為 22,035 次更新，總量接近原計畫，
但剩下只需約 153 次 epoch 結束驗證，而不是 1,499 次。
若仍指定 2,000 輪，剩下的總訓練更新量會放大約十倍；不能據此預期總耗時縮短。

對應既有中型模型存檔，可在原 .venv／PYTHONPATH 環境續訓：

~~~bash
python -m robot_bringup.train_act \
  --resume /mnt/d/robot_models/act_bottle_medium/20261021T005811Z_29e0cc88/last.pt \
  --samples-per-episode 10 \
  --epochs 654 \
  --validation-samples 1024 \
  --max-minutes 0 \
  --workers 2 \
  --device cuda:0
~~~

654 是依上述第 501 輪存檔估算，不是模型通用的建議輪數；應以實際存檔的 global_step
及期望總更新量調整。每輪耗時會增加，因為確實進行更多訓練，但單位更新對應的驗證次數會降低。
不需要重新編譯；新參數在下次啟動時套用，不會改變已執行的訓練程序。

## 中型設定與架構

| 項目 | medium 預設 |
|---|---|
| 可訓練／總參數 | **21,869,319／21,869,319**，約 2187 萬 |
| ResNet18 卷積參數 | 11,166,912，全部微調 |
| BatchNorm | FrozenBatchNorm2d：affine 與 running statistics 都固定 |
| 影像 | scene／wrist／side，共用一個 ImageNet ResNet18；每路原解析度 640×480 |
| Transformer hidden dimension | 256 |
| 視覺 encoder／動作 decoder／CVAE encoder | 4／4／4 層 |
| Attention heads／feedforward dimension | 8／1024 |
| Latent dimension／dropout | 32／0.1 |
| 動作 query／頻率 | 20／10 Hz，對應未來 2 秒 |
| Batch／梯度累積 | 4／2，有效 batch 約 8 |
| AdamW learning rate | Transformer 1e-5；ResNet18 1e-5 |
| Weight decay／KL weight | 1e-4／10 |
| 梯度裁切 | 預設關閉，與官方 optimizer 更新一致 |

「BN 固定」不代表凍結 ResNet18；卷積權重仍透過動作 loss 反向傳播更新。
實作參考 [官方 backbone](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/detr/models/backbone.py)。

- 使用 DETR 的 post-norm、ReLU，將位置資訊在每層 attention 加入 Q/K，V 保留內容。
- 動作 decoder 的起始內容全為零，20 個可學習 query 作為每層位置資訊。
- 各相機的 2-D sine 位置編碼獨立建立，特徵沿寬度串接；沒有額外的 camera embedding。
- CVAE 後驗使用 CLS＋當前 qpos＋action chunk，以遮罩排除 padding；由 mu/logvar 重參數取樣 z。
- 訓練與官方式驗證使用後驗；部署／額外 prior 指標使用 z=0，不讀取未來標籤。
- 使用最後一層 decoder 的輸出，確保全部 decoder 層接受訓練；不複製官方 release
  中 return_intermediate_dec=True 配上 hs[0] 只使用第一層輸出的行為。
- 三視角、7 維單臂與較小網路寬度／深度是本專案的縮放，並非原論文雙臂實驗的完整復現。

比較與例外見 [論文／官方程式對照](act_paper_audit.md)。

其他新架構 presets 也預設微調 CNN：

| 項目 | small | standard | medium |
|---|---|---|---|
| 可訓練參數 | 12,171,751 | 18,710,279 | 21,869,319 |
| hidden dimension | 128 | 256 | 256 |
| encoder／decoder／posterior | 1／2／1 | 2／4／2 | 4／4／4 |
| 影像寬×高 | 256×192 | 320×240 | 640×480 |

明確指定 CLI 尺寸優先於 preset；--freeze-backbone 僅用於凍結 CNN 的對照實驗。

## Loss、驗證與模型選擇

遵循 [官方 policy.py](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/policy.py)：

~~~text
L1 = mean(abs(prediction - action) * valid_mask)  # 分母包含 B×K×7
KL = mean_batch(sum_latent(KL(q(z|qpos,action) || N(0,I))))
loss = L1 + 10 × KL
~~~

padding 的誤差設成零，分母仍包含整個動作張量，與舊版只除有效步數的 loss 不同。
後驗 attention 同時遮罩 padding，不把無效標籤用於推斷 z。

驗證同樣使用 CVAE 後驗，best.pt 依 **val_posterior_loss（L1＋10×KL）** 選取。
另外單獨進行一次不給 action 的 z=0 forward，紀錄 val_prior_l1 與各關節 MAE（rad）。
prior L1 依有效元素計算，供部署誤差觀察，不用它取代官方的模型選擇目標。
驗證使用固定影格，不跟隨訓練的隨機 episode 抽樣。預設從整份驗證集的有效起點中，
以等距索引固定選取最多 **1,024 個時間點**；驗證樣本不足 1,024 時全部使用。
影像、qpos、未來最多 20 步標籤與讀取順序在每個 epoch 都相同，不會每次重抽。
這 1,024 筆是時間點，每個時間點有三視角影像，不是 1,024 個完整 episode。

驗證以 model.eval() 關閉 dropout，並在獨立 RNG context 中用相同 seed 抽 posterior z，
保留 CVAE 的高斯抽樣而不改成 z=mu。驗證結束或中斷後還原訓練 RNG，
避免驗證頻率影響後續訓練用的 dropout／z。相同權重與同一硬體設定下，
驗證可重複比較；這不是跨 CUDA／CPU／PyTorch 版本逐 bit 一致的保證。
驗證完整完成後才更新 best.pt。

預設每個 epoch 結束驗證。--validate-every-steps 可增加中途驗證；
--validation-samples 1024 為預設；--validation-samples 2048 可增加固定樣本數，
--validation-samples 0 依 episode 順序及各回合時間順序遍歷全部有效起點。
完整驗證會比以前每回合只抽一個時間點慢，仍會排除不合格觀測、跨資料缺口的動作與 padding 誤差。
--sampling frame 只改訓練資料的取樣方式，作為額外實驗；不改變驗證的固定取樣。

訓練啟動時會顯示 `Validation: 1024 fixed frames, sequential order, fixed posterior seed 42`
（實際數量取決於資料集）。config.json、metrics.jsonl、checkpoint 會記錄 validation_settings。
這是依本專案需求採用的固定驗證方式，與官方每次重新抽取驗證起點的流程不同；loss 公式不變。

### 既有 21 個驗證 demo 的實際數量

2026-10-21 依回合 `20261021T005811Z_29e0cc88/split.json`，用相同的 load_episode
時間對齊、執行區間與影像存在檢查統計：原始紀錄 7,566 個時間點；執行區間內 6,367 個；
可用驗證起點 **6,346 個**，對應三視角 **19,038 張輸入 RGB**。
其中 5,947 個起點有完整未來 20 步，其餘使用既有 padding mask，仍可合法驗證。
各 demo 的有效起點約 276～329 個；固定 256／1,024／2,048 筆分別占 4.03%／16.14%／32.27%。
這些比例是時間點涵蓋率；相鄰時間點和未來動作視窗會重疊，並非獨立新示範的數量。
完整遍歷這批驗證資料可指定 --validation-samples 0；增加驗證樣本主要增加每次驗證的時間，
batch size 與影像尺寸相同時不需一次將全部樣本放進 GPU。

## 資料與硬體上的調整

輸入為三路當前 RGB 與 7 維實測 qpos，關節順序為 joint_1～joint_6、
right_finger_bottom_joint，單位 rad。輸出第一列為 qpos(t+0.1s)，最後一列為 qpos(t+2s)。

保留使用者選擇的 future_measured_qpos 標籤，適合先以示範軌跡作基線，後續可銜接直接拖曳示教。
它與官方 leader 下達的目標關節命令不同，尤其可能存在追蹤延遲及夾爪接觸時的差距。
本次沒有改寫資料集或把幾何規劃路徑誤當成帶時間的 action。

- 只接受 schema 3、成功入箱、training_candidate=true、影像檢查通過的回合。
- 僅取執行期間，維持相機時間差 ≤20ms、因果關節狀態 ≤50ms、100ms±25ms 的連續片段。
- 跨 episode 或資料缺口不拼接動作序列。保留這些資料品質檢查。
- 預設 80%／20% 按 episode 切分；標準化只用訓練回合，保留 std 下限 0.01。
  官方以全體回合計算標準化，本專案保留 train-only 統計以隔離驗證資料。
- CUDA 預設 AMP（支援時 bf16，否則 fp16），並維持梯度累積 2；官方範例使用 FP32、batch 8。
  可加 --no-amp --batch-size 8 --accumulation-steps 1 比較；先確認顯存是否足夠。
- 累積時依實際 microbatch 樣本數加權，包含最後一個不足 batch 的更新。
- 這些工程調整不保證與官方實驗逐 bit 相同；也不保證單憑 loss 就能判斷抓取成功率。

若 8 GiB 顯存不足，可改 --batch-size 2 --accumulation-steps 4，保留有效 batch 約 8。
不必重新建環境；維持 NumPy <2 以相容 ROS cv_bridge。
第一次使用 ImageNet 權重需已有 torch cache 或可下載；--random-backbone 可另作隨機初始化實驗。

## 儲存、續訓與推理

每次新訓練建立獨立的時間戳目錄，保存 config.json、split.json、normalization.json、
metrics.jsonl、best.pt、last.pt 與 run_status.json；不覆寫既有訓練回合。

best.pt 是推理權重與設定，不含 optimizer；每次驗證改善時更新。
last.pt 保存完整續訓狀態，預設第一個 epoch、每 100 epochs、正常結束與時間預算到期時寫入，
避免在 D 槽每個短 epoch 都寫入完整 optimizer。可用 --save-every-epochs 調整；
明確開啟 step-based 中途驗證時，也會保存該驗證點的完整狀態。

~~~bash
python -m robot_bringup.train_act \
  --resume /mnt/d/robot_models/act_bottle_medium/實際新訓練目錄/last.pt \
  --epochs 4000 --max-minutes 0 --device cuda:0
~~~

續訓只接受 format 2，恢復模型、optimizer、scaler、資料切分、標準化與隨機狀態。
可覆寫 --epochs、--device、--workers、--max-minutes，其他參數以 checkpoint 設定為準。
--samples-per-episode 可覆寫每輪各訓練 demo 的抽樣次數；省略時沿用 checkpoint，舊存檔預設 1。
另外可覆寫 --validation-samples；省略時沿用新版 checkpoint 的固定驗證樣本數。
從舊的隨機驗證 format 2 checkpoint 續訓時，預設遷移到固定 1,024 個時間點。
若 checkpoint 已保存固定 256 筆等設定，續訓時加上 --validation-samples 1024 才會提高樣本數。
如果驗證設定改變，程式會先在新驗證資料上重新評估保留的 best.pt，再與目前權重比較；
不會直接拿舊的隨機驗證分數和新分數競爭。重新評分只更新驗證資訊，不更新該模型權重。
metrics.jsonl 可能同時保留新舊驗證紀錄，應依 validation_settings 區分。
部分 epoch 續訓會重新從該 epoch 開始取樣；多程序資料載入不保證逐 bit 重現。
Ctrl+C 保留最近已保存的 checkpoint；時間預算則會在 optimizer 邊界主動保存最新權重。

ACTPolicy.predict(images, current_qpos) 仍回傳 K×7 的 rad 絕對位置，部署不需未來標籤或最終姿態。
[Gazebo 推理程式](act_gazebo.md) 已同步檢查新版 checkpoint 格式。
Gazebo 推理使用即時觀測與 z=0，沒有從資料集隨機取樣的步驟；固定 VAL 不改變即時控制流程。
已經啟動的訓練程序不會因修改 Python 檔案而自動切換驗證方式，需要下次啟動或續訓才生效。

## 本次驗證

- 每輪多次抽樣更新後，全專案 **100 項測試通過**；確認各 demo 保持相同抽樣權重、
  每次重新選取起點、舊存檔可增加每輪更新量、固定 VAL 不變，且新抽樣數可保存並在續訓時還原。
  測試只使用合成資料，未啟動正式資料集訓練。
- 固定 VAL 更新後，全專案 **99 項測試通過**。新增確認固定時間點／順序、posterior 抽樣可重複、
  驗證不改動訓練 RNG 或模型參數、完整逐影格 VAL 不影響訓練的 episode 抽樣，
  以及驗證設定變更後重新評分保留的最佳模型。測試使用合成資料，未另開正式資料集訓練。
- 專案 87 項單元測試通過，包含 CNN 確實更新、BN buffers 不變、全部 decoder 層接受梯度、
  episode 抽樣、posterior 與 prior 分開驗證、posterior 選模、format 2 推理／續訓與較新 best.pt 保護。
- 與上述固定官方 commit 做數值比對：encoder、decoder 最大絕對差皆為 4.77e-7，
  FrozenBatchNorm 與 2-D sine 位置編碼差為 0；比較使用相同權重、dropout=0 的 FP32。
  可用 tests/check_act_reference.py --reference-dir <官方三個來源檔所在目錄> 重複比對。
- RTX PRO 1000 Blackwell Laptop、bf16、3×640×480、batch 4／累積 2，使用人造影像及關節資料
  完成 3 次 AdamW 更新。ImageNet 權重載入成功、CNN 更新、BN buffers 不變、梯度均有限。
- 該 GPU 測試 PyTorch 峰值配置 1770 MiB，峰值保留 2006 MiB（約 1.96 GiB），不包含其他程序用量。
  這是短時間合成資料測試，不是正式資料集完整訓練的耗時或整張 GPU 顯存用量保證。
- 未啟動正式資料集訓練、未覆寫舊模型、未執行 Gazebo 抓瓶評估。
