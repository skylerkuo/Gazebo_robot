---
license: apache-2.0
language:
- en
task_categories:
- robotics
tags:
- robotics
- simulation
- gazebo
- cable-insertion
- imitation-learning
- force-torque
- multiview
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/chunk-*/file-*.parquet
---

# AIC Cable Insertion — Gazebo (SFP0, Four Views)

I collected these trajectories in the [AIC Gazebo environment](https://github.com/intrinsic-dev/aic) for one task: inserting `sfp_tip` into `nic_card_mount_0/sfp_port_0`.

There are two types of episodes:

- **Direct:** the official CheatCode controller performs the whole insertion.
- **Assisted:** an ACT policy attempts insertion first. The official controller can then take over and correct the motion. The recording keeps both parts.

The starting point for this project was [ivogeorg/aic-cable-insertion-gazebo-v3](https://huggingface.co/datasets/ivogeorg/aic-cable-insertion-gazebo-v3). This dataset contains new recordings with an added side camera and labels showing which controller produced each action. The original dataset files are not included.

{{UPDATE_NOTE}}
## Dataset contents

| Item | Value |
|---|---|
| Episodes | {{EPISODES}}, all with successful official insertion scores |
| Frames | {{FRAMES}} |
| Sampling rate | 20 Hz, measured in simulation time |
| Direct episodes | {{DIRECT}} |
| Assisted episodes | {{ASSISTED}} |
| Assisted episodes with complete expert windows | {{RECOVERY}} |
| Cameras | Left, center and right wrist cameras, plus a fixed side camera |
| Images | 576 × 512 RGB; PNG bytes stored inside Parquet |
| Robot | UR5e |
| Action | Absolute TCP target `[x, y, z, qx, qy, qz, qw]` in `base_link` |
| Parquet size | {{GIB}} GiB |
| License | Apache-2.0; see `LICENSE` and `NOTICE` |

All episodes were collected automatically in simulation. There are no human demonstrations. Some ACT attempts succeeded before expert takeover, so an assisted episode does not always contain a complete correction window. A successful final outcome also does not mean every earlier ACT action was correct.

## Download and read an episode

Each Parquet file contains one episode, including all four camera streams. To preview the data, download one file rather than the full dataset:

```python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # HTTP compatibility mode for this preview

from huggingface_hub import hf_hub_download
from datasets import load_dataset
import pyarrow.dataset as pa_dataset

path = hf_hub_download(
    "Kuoskyler/aic-cable-insertion-gazebo-sfp0",
    filename="data/chunk-000/file-000.parquet",
    repo_type="dataset",
    local_dir="./aic_sfp0_preview",
    token=False,
)
ds = load_dataset(
    "parquet",
    data_files=path,
    split="train",
    streaming=True,
    batch_size=1,
    fragment_scan_options=pa_dataset.ParquetFragmentScanOptions(pre_buffer=False),
)
iterator = iter(ds)
try:
    frame = next(iterator)
    print(frame["episode_index"], frame["action_source"], frame["action"])
    frame["observation.images.side_camera"].save("side_camera.png")
finally:
    iterator.close()
```

This example reads a local file and closes the iterator explicitly. Remote Parquet streaming caused shutdown problems in some short Python preview runs during testing.

Use `snapshot_download` with `allow_patterns` to download selected files. `meta/episodes.jsonl` lists episode IDs and the `episode_type` (`direct` or `assisted`).

```text
data/chunk-000/file-*.parquet         One episode per file
meta/info.json                      Feature names, shapes and sampling rate
meta/tasks.parquet                  Task description
meta/episodes.jsonl                  Episode type, scores, controller labels and window indices
meta/release.json                    Counts and collection versions
meta/snapshot.json                   Episode IDs and Parquet checksums
meta/files.json                      File sizes and SHA-256 hashes
calibration/episode-*.json           Camera intrinsics and frame transforms
environment/scenes/episode-*.json    Board and cable configuration for each episode
environment/                        AIC assets and four-view setup files
```

The Hugging Face `train` split holds the entire dataset. Make your own Train/VAL split by whole episode, and compute normalization from Train only. Nearby frames and overlapping action chunks from one episode should stay in the same split.

The files use LeRobot-style feature names but are a custom Parquet export. Read them with Hugging Face Datasets or PyArrow; they have not been validated as a native LeRobot v3 dataset. Policy checkpoints, ROS bags and training caches are not included.

## Observations and actions

| Column | Contents |
|---|---|
| `observation.images.{left,center,right,side}_camera` | RGB images with embedded PNG bytes |
| `observation.state` | Six values: elbow, left gripper finger, shoulder lift, shoulder pan, wrist 1, wrist 2 |
| `observation.arm_joint_positions` | Six arm joints: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3; radians |
| `observation.environment_state` | Wrist wrench `[Fx,Fy,Fz,Tx,Ty,Tz]`; N and N·m |
| `observation.tcp_pose` | Measured TCP pose `[x,y,z,qx,qy,qz,qw]` in `base_link` |
| `action` | Commanded TCP target in the same order; metres and unit quaternion |
| `action_source` | `model`, `expert` or `hold` |
| `action_segment` | Controller segment ID |
| `human_active` | Legacy flag, also true during automatic expert control |
| `episode_index`, `frame_index`, `index`, `task_index` | Episode, frame, row and task IDs |
| `timestamp`, `simulation_time` | Relative episode time and simulation clock |
| `action_time`, `joint_time`, `wrench_time` | Command, joint-state and wrench timestamps |
| `camera_times` | Image timestamps in left, center, right, side order |

`observation.state` includes a gripper joint and omits wrist 3. Use `observation.arm_joint_positions` if you need all six arm joints. Wrench values are stored as recorded; the sensor frame is defined by the calibration and simulator files. Do not assume gravity compensation.

## How the data was collected

The CheatCode controller uses the simulator's plug, port and TCP transforms to align and insert the connector. The ACT policy receives images, joint state and wrench; it does not receive the plug or port ground-truth pose.

Board position and yaw vary between runs, though some early episodes use the nominal scene. Use the scene and calibration files for the episode you are reading. The representative environment files and collection defaults do not describe every historical run exactly.

An episode was accepted only after successful official insertion scoring, a successful policy return and no recorder writer error. Scores and diagnostic insertion events are stored in the episode metadata. Event times are callback receipt times, so they are approximate markers rather than exact contact times.

## Using the correction data

The recordings keep the ACT attempt, expert takeover and stationary portions. Filtering is recorded through window indices and masks; separated time intervals are not joined together.

At 20 Hz, a 20-step training chunk anchored at frame `t` uses actions `t+1` through `t+20`. An expert-only chunk must contain continuous expert actions within one control segment, with valid timestamps. The metadata includes `expert_anchor_frames`, `filtered_expert_anchor_frames` and `expert_validation_frames`. Recheck these windows if you change the horizon or sampling rate.

The idle filter downsamples long windows where both commands and wrench barely change, while keeping transition boundaries. Little motion alone is not enough to discard contact data. The alternative rollout filter keeps moving ACT targets and masks long stationary model targets; its indices and reports are also included. Choose the filter that fits your training objective, and keep `model`, `expert` and `hold` labels separate.

## Known limitations

- This is one simulated connector/port task with modest scene variation. It has not been evaluated on a physical robot.
- Only successful episodes are included. The ACT portion may still contain mistakes before recovery.
- Actions are command targets, not measured TCP positions. Under contact, they can differ considerably. In some direct episodes, the controller keeps lowering the Z target after insertion while the measured TCP barely moves. Those frames remain in the files.
- Low validation error against these commands does not establish a high insertion success rate. Test the trained policy in closed loop as well.
- The asset archive contains AIC source packages and collection configuration. You still need the official ROS 2/Gazebo environment and its dependencies; see `environment/README.md`.

## Credits and citation

Collected and published by Kuoskyler. Please cite this dataset, the original community dataset and the official AIC project when using this work.

- Original community dataset: [ivogeorg/aic-cable-insertion-gazebo-v3](https://huggingface.co/datasets/ivogeorg/aic-cable-insertion-gazebo-v3), revision `9d2ac51be268ef27cf2c2d0e073c7d9bb9acbe2c`.
- Simulation and controller: [Intrinsic AIC](https://github.com/intrinsic-dev/aic), revision `6546127c1c15d5786f59c91400f5fc5ad6894448`.

Upstream copyright and license notices are retained in the bundled assets and `NOTICE`. The four-view camera and world files are collection additions. This is a community dataset, not an official AIC release.

```bibtex
@misc{kuoskyler_aic_sfp0_2026,
  author = {Kuoskyler},
  title = {AIC Cable Insertion -- Gazebo (SFP0, Four Views)},
  year = {2026},
  howpublished = {Hugging Face dataset},
  url = {https://huggingface.co/datasets/Kuoskyler/aic-cable-insertion-gazebo-sfp0}
}
@misc{ivogeorg_aic_cable_insertion,
  author = {ivogeorg},
  title = {AIC Dexterous Cable Insertion -- Gazebo (LeRobot v3)},
  howpublished = {Hugging Face dataset},
  url = {https://huggingface.co/datasets/ivogeorg/aic-cable-insertion-gazebo-v3}
}
@misc{intrinsic_aic,
  author = {{Intrinsic}},
  title = {AI for Industry Challenge (AIC)},
  howpublished = {Software repository},
  url = {https://github.com/intrinsic-dev/aic}
}
```
