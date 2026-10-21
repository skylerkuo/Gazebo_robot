# ACT 論文與目前實作對照審查

## 2026-10-21 新版中型架構

本專案已改為 format 2：ResNet18 全卷積微調、FrozenBatchNorm2d、DETR post-norm／ReLU、
每層 Q/K 位置編碼、零起始 decoder 內容、沿寬度拼接相機特徵、CVAE 後驗訓練與部署 z=0。
medium 為 hidden 256、encoder／decoder／posterior 各 4 層、FF 1024，21,869,319 個可訓練參數。

訓練採用每個 episode 隨機一個起點、AdamW、L1＋10×KL；padding 分母與官方 policy.py 相同。
驗證以 posterior L1＋KL 選 best.pt，另記錄不讀未來 action 的 prior L1／MAE。
依使用者要求，驗證改成固定影格、固定讀取順序、固定 posterior 隨機 seed：
預設 1,024 個時間點，--validation-samples 0 則依序遍歷所有有效時間點。
訓練仍保留每個 episode 隨機起點；Gazebo 仍使用即時觀測與 z=0。
可用 --samples-per-episode 10 增加每輪各 demo 的隨機起點數，降低驗證相對頻率；
預設 1 保留原規模。這會改變 epoch 的訓練量，應依 global_step 比較總更新量，不能直接沿用輪數估時。
預設學習率 1e-5，backbone 1e-5；batch 4／累積 2，AMP 為 8 GiB GPU 的工程調整。
舊版 format 1 依使用者要求不再相容。設定、指令與差異詳見 [新版訓練文件](act_training.md)。

對照官方 commit：742c753c0d4a5d87076c8f69e5628c79a8cc5488。
保留的差異包括三相機／7 維、10 Hz／20 步、較小網路、未來實測 qpos 標籤、
train-only 標準化、固定驗證資料／posterior 抽樣與額外 prior 指標；驗證在 epoch 結束後進行。
採用最後 decoder 層，不重現官方 release 取 hs[0] 導致只訓練第一層輸出的問題。
這是依官方核心實作縮放的模型，並非原論文資料、硬體或實驗的完整復現。

Gazebo 推理同步使用同一 ACT 類別、checkpoint 的影像尺寸／相機順序／訓練統計、
最後 decoder 層與 z=0。CUDA 可依 checkpoint 的 AMP 設定選 bf16／fp16，亦可指定 FP32。
推理沒有 optimizer 或梯度累積；細節與實際驗證見 [Gazebo 執行文件](act_gazebo.md)。

## 以下為 2026-10-20 舊版審查紀錄

以下「目前」與參數描述均指當日舊版程式，不是上述新版架構。

審查日期：2026-10-20。範圍為目前原始碼與 `--preset small`，不代表正在執行的程序
一定採用最新預設；已啟動的程序／checkpoint 以其 `config.json` 為準。

結論：目前是保留 ACT 核心 CVAE／action chunking 的獨立精簡變體，不能稱為原論文或
作者程式的完整復現。符合核心架構與一字不差復現實驗，是不同的判定。
本次只審查及補文件，未修改模型、動作標籤、checkpoint，也未中斷或另開正式訓練。

## 對照來源

優先使用原論文與原作者程式，不將教學文章當作規範：

1. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/pdf/2304.13705)：
   第 IV 節、圖 4／5、Algorithms 1／2、Appendix C 與 Table III。
2. [官方模型 detr_vae.py](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)。
3. [官方 Transformer](https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py)。
4. [官方 backbone](https://github.com/tonyzhaozh/act/blob/main/detr/models/backbone.py)。
5. [官方 policy／loss](https://github.com/tonyzhaozh/act/blob/main/policy.py)。
6. [官方資料處理](https://github.com/tonyzhaozh/act/blob/main/utils.py)、
   [模擬示範收集](https://github.com/tonyzhaozh/act/blob/main/record_sim_episodes.py)。
7. [官方訓練／推論](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)、
   [optimizer](https://github.com/tonyzhaozh/act/blob/main/detr/main.py)、
   [官方 README](https://github.com/tonyzhaozh/act/blob/main/README.md)。

官方連結為審查當時的公開 main 分支，未聲稱其與論文每項細節完全一致。

## 主要發現，依影響排序

### 1. 監督標籤的語意與原方法不同

位置：`robot_bringup/act_data.py` 的 `ACTION_SOURCE`、`Episode.chunk()`。

目前資料集沒有獨立、同步的七維 command action 欄位，程式實際使用：

```text
輸入：images(t), q_measured(t)
標籤：q_measured(t+dt), ..., q_measured(t+K*dt)
dt = 0.1 s
```

論文第 IV 節明確區分示範者給的目標命令與 follower 的實測狀態；兩者的差值會影響低階控制。
原作者模擬資料也會以命令重播產生觀測，並特別保留夾爪控制目標，見 `record_sim_episodes.py`。

目前學習的是未來實測狀態的模仿基準。這是有用的實驗，但不等於原 ACT 的 command action
behavior cloning。接觸物體後，實測位置可能停止變化，而命令仍維持較大的閉合目標；
因此推論時只追蹤預測實測位置，可能無法重現原控制命令形成的夾持效果。
這是控制語意的差異，不應用「把索引 t+1 改為 t」草率處理。

實際抽查既有成功回合：

```text
/mnt/d/robot_datasets/bottle_pick_place_rgb/20261020T033300Z_1e7861db/episode_0000
solution.json 夾爪規劃目標最大值：約 0.700000 rad
frames.jsonl 夾爪實測位置最大值：約 0.666669 rad
```

這只是一回合的極值比較，未做逐時刻命令對齊，不以此推估夾持力或所有回合的差異。
bag 確實有手臂 controller state 的 `reference`、`feedback` 欄位；抽樣訊息的 reference 包含六軸位置。
夾爪目前記錄 feedback/status，並有預先規劃 solution，但不能假設它們已形成與 RGB 精確對齊的
完整實際命令序列。是否能從既有資料完整重建，還需另行驗證；缺少的命令不可憑空當作已記錄。

### 2. 完整的閉迴路推論流程尚未交付

位置：`robot_bringup/act_policy.py` 的 `ACTPolicy.predict()`。

目前一次輸出 K×7，沒有跨呼叫保存重疊 chunks 的緩衝區，也沒有把同一未來時刻的多個預測
做 temporal ensembling；目前也沒有 ROS 策略執行節點。
論文 Algorithm 2 與作者 `imitate_episodes.py` 包含時間對齊、重疊預測集成與環境動作套用。

這不會讓離線訓練失去效力，temporal ensembling 本來就是推論端處理；但不能把目前的
`predict()` 當作已完成整套 ACT 控制系統。現在也尚無模型閉迴路成功率可據以宣稱抓放能力。

### 3. Transformer 實作的差異超過層數縮小

位置：`robot_bringup/act_model.py` 的 encoder 建立函式及 `forward()`。

目前使用 PyTorch 內建 Transformer，GELU、pre-norm，將空間／相機位置編碼一次加入輸入；
decoder 直接把可學習 query embedding 當作初始 tgt 內容。

原作者使用 DETR 式自訂 Transformer：位置／query 編碼在各 attention 層的 Q/K 路徑加入，
decoder 的初始 tgt 為零，query_pos 與 tgt 分開；公開預設使用 ReLU，未指定 pre_norm 時為 post-norm。
見官方 `transformer.py`、`detr/main.py`。

因此目前具有 ACT 的 encoder–decoder 結構，但不是「原始 attention block 只縮小維度」。
若要結構忠實的輕量版，應保持小維度，另行實作相同的 positional/query 注入方式。
這類修改應使用新架構版本及新訓練，不能靜默套用到已產生的 checkpoint。

注意來源本身的差異：論文用固定 decoder 位置嵌入描述架構，作者程式則使用可學習
`query_embed`；不能只因目前的 query 是 learned embedding 就判定不符合作者實作。
作者公開程式另有 `return_intermediate_dec=True` 配合 `hs[0]` 選擇第一層輸出的行為，
也不應把它誤當成論文要求複製的設計。

### 4. 預設凍結整個 CNN，與作者公開訓練不同

位置：`act_model.py` 的 backbone `requires_grad_`／`train()`／`torch.no_grad()`。

目前 ResNet18 完全凍結，只訓練 projection、Transformer、CVAE 等部分，small 預設可訓練參數約 101 萬。
作者 `backbone.py` 凍結的是 BatchNorm，CNN 卷積參數仍可參與訓練；optimizer 另設 backbone learning rate。
「凍結 BatchNorm」和「凍結整個 ResNet」不相同。

這是為縮短時間／減少顯存採取的取捨，可能限制對目前相機視角與小瓶子的特徵適應。
現有 `--train-backbone` 可解除整個 CNN 的參數凍結，但本實作的 BN affine 與官方 FrozenBatchNorm
仍有差異，開啟該旗標不能單獨讓整個模型變成忠實復現，也未在本次變更正在進行的訓練。

### 5. 驗證／checkpoint 選擇不是官方的原始指標

位置：`train_act.py` 的 `validate()`、`validation_subset()`、`evaluate_and_save()`。

目前驗證呼叫 `model(qpos, images)`，以 z=0 的 normalized prior L1 選 best；small 預設只評估固定 256 筆
held-out 觀測。官方驗證會把 action 傳入 CVAE posterior，並以重建 L1＋βKL 的總 loss 選 best，
見 `imitate_episodes.py` 的 `forward_pass()`／`train_bc()` 及 `policy.py`。

目前的 prior 指標有助於觀察部署條件的預測誤差；官方 posterior 驗證也是合法的 CVAE 驗證方法。
兩者不是同一個評估量，不能把本專案的 `0.1` 與他人的 posterior／總 loss 直接比較。
較完整的報告應同時保留 posterior objective、prior L1、各關節實際誤差與獨立閉迴路成功率。
目前驗證集也同時用於 checkpoint 選擇，尚無獨立 test rollout 結果。

## 已符合的核心設計

以下判定依官方 `detr_vae.py`／`policy.py` 與目前程式逐項核對：

| 核心項目 | 目前實作 | 判定 |
|---|---|---|
| 觀測 | 同一組 RGB 與當前 qpos，含夾爪 | 符合核心；7 維是單臂適配 |
| Action chunk | 一次輸出 K×7 絕對位置，不是逐步自回歸 | 符合序列預測設計；標籤語意見發現 1 |
| CVAE posterior | CLS＋當前 qpos＋示範序列，無 RGB | 符合 |
| 後驗分布 | 對角 Gaussian 的 μ、logvar 與重參數化取樣 | 符合；額外 logvar clamp 為數值穩定措施 |
| Conditional policy | 視覺特徵、qpos、z 經 encoder，再由 decoder 預測 | 符合大架構；attention 細節見發現 3 |
| 推論 latent | 不輸入真實未來動作，使用 z=0 | 符合原方法，無須改掉 |
| 重建損失 | 有效位置的 absolute L1 | 與作者使用 L1 的原則一致 |
| KL | 對 q(z|qpos, actions) 與 N(0,I) 計算 KL，對 latent 維度加總 | 符合 |
| 合成 loss | L1＋βKL，預設 β=10 | 符合主要目標 |
| 輸入／輸出尺度 | qpos/action mean-std，輸出還原為物理位置 | 與官方處理原則相同 |
| optimizer | AdamW 與分開的 backbone learning rate | 與作者程式類型一致 |

論文 Algorithm 1 偽碼的 reconstruction 標成 MSE，但正文 IV-C 明確說使用 L1，作者
`policy.py` 也實際使用 L1。因此這裡不應為追隨該偽碼單行而改成 MSE。

## 已調整的超參數與資料流程

參數是可調設計，改成單臂／較低 FPS 並不自動否定 ACT 核心方法，但應公開記錄：

| 項目 | 論文／作者常用設定 | 目前 small |
|---|---|---|
| 影像與狀態 | 4 視角／雙臂 14 維 | 3 視角／單臂＋夾爪 7 維 |
| 動作頻率與 chunk | 50 Hz、K=100 | 10 Hz、K=20；同為 2 秒視窗但控制頻率不同 |
| hidden／FFN／latent | Table III／圖示：512／3200／32 | 128／512／16 |
| encoder／decoder／posterior | 論文圖示：4／7／4 | 1／2／1；不聲稱複製官方中間輸出行為 |
| learning rate | Table III／README：1e-5 | 1e-4 |
| batch | 論文 8 | 最新 small 為 16，累積 2 次，即通常每次更新 32 筆 |
| 時間／訓練量 | 官方以長時間訓練及 rollout 評估 | 使用者要求時間預算，不保證跑滿 epoch 上限 |

`utils.py` 的原始 loader 以回合為 dataset item，取該回合的隨機 t；目前 loader 以所有有效影格為 item，
每個 epoch shuffle 一次。較長回合在目前資料流程中有較高權重；這是取樣方法差異，不是 CVAE 架構錯誤。
兩邊 epoch 的更新步數差很多，不應只用「原作者 2000 epochs、我們 3 epochs」判斷訓練量。

其他可接受但與原版不完全相同的工程處理：

- Padding：目前 L1 除以有效 action 元素數；官方原始版本先將 padding 誤差設零，再對全部元素 mean。
  尾端的梯度尺度因此不同，兩者都沒有直接把 padded target 當真實動作訓練。
- 本專案只用 training episodes 計算標準化，並檢查相機／狀態同步及斷幀；不需為追求一致而移除。
- AMP、gradient accumulation、clip grad norm=1、時間預算與 checkpoint 是額外工程設定。
- `train L1`／`KL` 是該 epoch 截至當下的累積平均；`val prior L1` 是當下權重的驗證。
  因此 epoch 切換造成 train 數字跳動，或 train 大於 val，都不能單憑數字差宣稱過擬合／程式錯誤。

## 建議處理順序

1. 保留本次模型與資料作為明確標記的 future-measured-qpos baseline，不中途改動同一 run 的 supervision。
2. 優先確認／建立包含夾爪的七維實際命令監督，嚴格驗證 command、RGB 與 qpos 的時間對齊。
3. 補上閉迴路推論、chunk 時間對齊與 temporal ensembling，再量測未見起始位置的任務成功率。
4. 若需要較忠實的 ACT 變體，在保留小維度的同時改用 DETR 式位置／query 注入，
   並明確選擇 CNN 微調策略；使用新版本 checkpoint，重新訓練比較。
5. 同時記錄 prior 與 posterior 指標；不要把公開模型的異質 loss 當成通用門檻。

## 本次驗證與限制

重新執行 `tests/test_act_training.py`：20 項通過，包含形狀、posterior padding、有限梯度、
未來標籤索引、回合切分、checkpoint round-trip、時間預算及部分 epoch 續訓。
這些測試驗證本實作的行為，並不是與官方模型逐層數值等價的測試。

另以唯讀方式抽查一回合的 metadata、frames、solution 與 bag controller 訊息。
本次沒有重訓、改變正在執行的模型、完整重建所有命令、量測接觸力或執行閉迴路控制。
因此不能從本次審查推斷實際成功率，也不能說現有 checkpoint 已完成原 ACT 實驗復現。
