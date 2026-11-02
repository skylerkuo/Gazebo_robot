# AIC Gazebo environment reference

The dataset was collected in the official AIC Gazebo environment:

- Source: https://github.com/intrinsic-dev/aic
- Source revision: `6546127c1c15d5786f59c91400f5fc5ad6894448`
- Official runtime: `ghcr.io/intrinsic-dev/aic/aic_eval@sha256:253b99a4aa6e0e4fc0cf8c1cfd0269e64aa86411c2bd585c95a1732d4a56a165`
- Upstream license: Apache-2.0. Copyright notices in the asset source packages are preserved.

`aic_assets_and_description.tar.gz` contains the upstream `aic_assets` and `aic_description` source packages and upstream LICENSE. It includes AIC board, connector, camera mount, force sensor and gripper assets. It is not the complete runtime, and UR robot description, ROS 2/Gazebo plugins and other dependencies must be obtained from the official environment. Follow the upstream installation/build instructions at the pinned source revision. The archive does not contain learned policy weights.

To inspect the asset source packages:

```bash
mkdir aic_source_assets
tar -xzf aic_assets_and_description.tar.gz -C aic_source_assets
```

`sideview_world.sdf`, `recording_camera.xacro`, `recording_robot.xacro` and `bridge.yaml` are representative collection configuration files for the added four-view setup. They are reference files, not a standalone launch command. The robot Xacro's generated absolute include of the recording camera was changed to the relative `recording_camera.xacro`; keep both files together. Other package-relative references need a configured ROS package environment. These collection configuration changes are separate from the unchanged upstream package archive.

`scenes/episode-*.json` gives each accepted episode's task-board and cable configuration. `../calibration/episode-*.json` supplies recorded camera intrinsics, camera/frame transforms and side-view geometry. Use these files for the particular episode rather than treating the representative configuration as universal.

The task is `sfp_tip` into `nic_card_mount_0/sfp_port_0`. Collection defaults record at 20 Hz and 576×512 RGB. The side camera is fixed near world position `[0.60, -0.60, 1.42]`, looking toward `[0.15, -0.20, 1.27]`. Scene variation defaults allow board offsets up to 8 mm in X/Y, 2 mm in Z and 0.02 radians in yaw; exact historical scenes are retained per episode.

The CheatCode demonstrator uses privileged simulator transforms for plug/port alignment. A learned policy consuming images, joints and wrench does not automatically have this information.
