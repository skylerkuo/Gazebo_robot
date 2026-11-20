# Public SFP0 configuration templates

These templates cover four-view data collection, ACT training, frozen-ACT Residual Flow training, and Gazebo inference. Start in the repository root with the Python and Gazebo environment described in [Environment](ENVIRONMENT.md). Set `SFP0_STORAGE` to a writable directory on your WSL D: mount. The existing launchers use that drive for model and evaluation outputs.

## 1. Prepare local settings

The renderer substitutes only whole-field path placeholders. Settings such as batch size, seed, and ODE steps remain ordinary JSON values. Your generated files can go in the Git-ignored `robot_insert/configs/local/` directory.

```bash
: "${SFP0_STORAGE:?Set SFP0_STORAGE to your writable experiment directory}"
CFG=robot_insert/configs/local

python robot_insert/scripts/render_sfp0_public_config.py collection \
  --output "$CFG/collection.json"

python robot_insert/scripts/render_sfp0_public_config.py act-train \
  --set "ACT_DATASET=$SFP0_STORAGE/direct" \
  --set "ACT_OUTPUT=$SFP0_STORAGE/models/act" \
  --output "$CFG/act-train.json"

python robot_insert/scripts/render_sfp0_public_config.py flow-train \
  --set "RECOVERY_DATASET=$SFP0_STORAGE/recovery" \
  --set "ACT_CHECKPOINT=$SFP0_STORAGE/models/act/checkpoints/best.pt" \
  --set "FLOW_OUTPUT=$SFP0_STORAGE/models/flow" \
  --output "$CFG/flow-train.json"

python robot_insert/scripts/render_sfp0_public_config.py act-gazebo \
  --output "$CFG/act-gazebo.json"
python robot_insert/scripts/render_sfp0_public_config.py flow-gazebo \
  --output "$CFG/flow-gazebo.json"
```

`collection.json` is a camera and scene profile for `collect_sfp0_sideview.py --config`. The ACT trainer expects direct demonstrations in the local collection format, including the successful episode records. The Flow trainer expects expert corrections in the assisted collection format and a compatible ACT checkpoint. The public Hugging Face export uses a distribution schema. If using an export in place of local collection, prepare the manifest and index required by the selected trainer first. For local demonstrations, the entry points are `collect_sfp0_direct.py --output-dir` for direct trials and `collect_sfp0_expert.py --checkpoint --output-dir` for assisted trials.

## 2. Train

Once the corresponding dataset and checkpoint exist, run the readiness check first. The default validation budget is 1,500 sampled windows, so the validation episodes must contain at least that many valid windows.

```bash
python robot_insert/scripts/train_sfp0_act_multiview.py \
  --config "$CFG/act-train.json" --check-data

python robot_insert/scripts/train_sfp0_act_multiview.py \
  --config "$CFG/act-train.json"

python robot_insert/scripts/train_sfp0_residual_flow_multiview.py \
  --config "$CFG/flow-train.json" --check-data

python robot_insert/scripts/train_sfp0_residual_flow_multiview.py \
  --config "$CFG/flow-train.json"
```

ACT initializes its visual backbone with ImageNet weights. The Flow template prepares an index from the supplied recovery collection and uses the ACT checkpoint's normalization. Continue a stopped run with that trainer's `--resume` and its own `checkpoints/last.pt`.

## 3. Evaluate with official scoring

Choose your own seed range and a fresh output directory. The example runs headless for 100 trials with a 75-second simulation-time limit per trial.

```bash
python robot_insert/scripts/evaluate_sfp0_residual_flow.py \
  --checkpoint "$SFP0_STORAGE/models/flow/checkpoints/best.pt" \
  --output-dir "$SFP0_STORAGE/evaluations/flow" \
  --scene-start 10001 --trials 100 --gui false \
  --real-time-factor 0.2 --max-task-seconds 75 --max-wall-seconds 900
```

The evaluator saves `summary.json`, `episodes.csv`, and per-trial result files. Resume with the same `--output-dir` and `--resume`. For one scene or the Gazebo UI, use `run_sfp0_multiview_gazebo.py` with `--config "$CFG/flow-gazebo.json"` (or the ACT config), an explicit `--checkpoint`, and `--results-dir`.
