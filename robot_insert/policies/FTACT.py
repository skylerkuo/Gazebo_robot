"""Learned RGB/state/wrench policy for the official AIC InsertCable interface."""
from contextlib import nullcontext
from collections import deque
import json
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
import torch
from aic_model.policy import Policy
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from std_msgs.msg import Header

from ftact import load_checkpoint
from ftact_control import (STATE_JOINT_NAMES, bounded_pose, observation_arrays,
                           pose_array, stamp_seconds, unit_pose, TimedActionChunk,
                           ACTTemporalPoseEnsembler)
from multiview_control import CAMERA_KEYS,multiview_arrays,ObservationNotReady


class FTACT(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.settings = json.loads(Path(os.environ["FTACT_SETTINGS"]).read_text())
        self.output = Path(self.settings["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(2)
        self.device = self.settings["device"]
        self.model, payload = load_checkpoint(self.settings["checkpoint"], self.device)
        self.multiview = payload['format'] in ('aic-sfp0-act-multiview-v1','aic-sfp0-act-pace-multiview-v1','aic-sfp0-act-pace-pred-multiview-v1','aic-sfp0-flow-matching-multiview-v1','aic-sfp0-supervised-residual-flow-multiview-v1','aic-sfp0-stacked-residual-flow-multiview-v1','aic-sfp0-alternating-act-residual-flow-multiview-v1')
        self.camera_names = ('left','center','right','side') if self.multiview else ('left','center','right')
        self.flow_matching = payload['format'] in ('aic-sfp0-flow-matching-v1','aic-sfp0-flow-matching-multiview-v1','aic-sfp0-supervised-residual-flow-multiview-v1','aic-sfp0-stacked-residual-flow-multiview-v1','aic-sfp0-alternating-act-residual-flow-multiview-v1')
        self.pace = payload['format'] in ('aic-sfp0-act-pace-v1','aic-sfp0-act-pace-multiview-v1','aic-sfp0-act-pace-pred-multiview-v1')
        single_task = payload['format'] == 'aic-sfp0-act-v2' or self.flow_matching or self.pace or self.multiview
        self.pace_context = None
        if self.pace:
            from pace_act import TactileContext
            self.pace_context = TactileContext(payload['config']['phase_alpha'], payload['config']['tactile_history'])
        mode = self.settings.get('execution_mode', 'first_action')
        self.act_receding = single_task and not self.flow_matching and mode == 'sfp0_act_receding_horizon'
        self.timed_chunks = single_task and mode in ('sfp0_timed_chunks', 'sfp0_flow_receding_horizon',
                                                    'sfp0_act_receding_horizon')
        self.temporal_ensemble = single_task and not self.flow_matching and mode == 'sfp0_act_temporal_ensemble'
        self.task_contract = payload.get('task')
        if single_task:
            if self.flow_matching != (mode == 'sfp0_flow_receding_horizon'):
                raise ValueError('Flow Matching checkpoints require the Flow Matching inference config')
            if not (self.timed_chunks or self.temporal_ensemble) or self.settings['rate_hz'] != 20:
                raise ValueError('v2 checkpoint requires the SFP0 v2 20 Hz inference config')
            if payload.get('action_offset') != 1 or payload['config']['chunk_size'] != 20:
                raise ValueError('v2 inference requires the next-20-actions training contract')
            if mode == 'sfp0_timed_chunks' and self.settings.get('n_action_steps') != payload['execution']['n_action_steps']:
                raise ValueError('Configured replanning horizon differs from v2 checkpoint')
            if self.temporal_ensemble and self.settings.get('n_action_steps') != 1:
                raise ValueError('ACT temporal ensembling must query on every control step')
            if self.multiview:
                if not (self.temporal_ensemble or self.flow_matching) or payload.get('camera_order')!=list(CAMERA_KEYS):
                    raise ValueError('Four-view policy requires its camera order and matching execution config')
                if self.settings.get('camera_order')!=list(CAMERA_KEYS):
                    raise ValueError('Inference camera order differs from the four-view checkpoint')
            if self.pace and not self.temporal_ensemble:
                raise ValueError('PACE uses Event-corrected chunks with ACT temporal ensembling')
            if self.act_receding and self.settings.get('n_action_steps') != 1:
                raise ValueError('The ACT comparison baseline must replan every control step')
            if self.flow_matching:
                if not 1 <= self.settings.get('n_action_steps', 0) <= 20:
                    raise ValueError('Flow Matching n_action_steps must be between 1 and 20')
                self.model.configure_sampling(steps=self.settings['ode_steps'], solver=self.settings['ode_solver'],
                                              seed=self.settings['noise_seed'])
        elif mode != 'first_action':
            raise ValueError('The SFP0 v2 inference config requires a v2 checkpoint')
        self.checkpoint_step = payload["step"]
        self.minimum = np.asarray(payload["stats"]["action"]["min"][:3]) - self.settings["workspace_margin_m"]
        self.maximum = np.asarray(payload["stats"]["action"]["max"][:3]) + self.settings["workspace_margin_m"]
        if tuple(self.settings["state_joint_names"]) != STATE_JOINT_NAMES:
            raise ValueError("This checkpoint adapter requires the verified six-joint source order")
        if self.settings["action_frame"] != "base_link":
            raise ValueError("The AIC pose dataset uses absolute TCP poses in base_link")
        if self.settings["amp"] not in ("none", "bf16", "fp16"):
            raise ValueError("Invalid inference precision")
        if self.device == "cpu" and self.settings["amp"] != "none":
            raise ValueError("CPU inference requires amp=none")
        if self.settings["amp"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This device does not support BF16")
        self.journal = (self.output / "inference.jsonl").open("a", buffering=1)
        # The engine cleans up/reconfigures the policy between trials. Keep
        # one subscription and task numbering on the node across those resets.
        if not hasattr(parent_node, "_ftact_source_joints"):
            history = deque(maxlen=500)
            parent_node._ftact_source_joints = history
            parent_node._ftact_joint_subscription = parent_node.create_subscription(
                JointState, "/joint_states", lambda message: history.append(message),
                qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())
        self.source_joints = parent_node._ftact_source_joints
        if self.multiview:
            from sensor_msgs.msg import Image,CameraInfo
            if not hasattr(parent_node,'_ftact_side_images'):
                parent_node._ftact_side_images=deque(maxlen=100)
                parent_node._ftact_side_info=None
                parent_node._ftact_side_subscription=parent_node.create_subscription(
                    Image,'/side_camera/image',lambda message:parent_node._ftact_side_images.append(message),
                    qos_profile_sensor_data,callback_group=ReentrantCallbackGroup())
                parent_node._ftact_side_info_subscription=parent_node.create_subscription(
                    CameraInfo,'/side_camera/camera_info',lambda msg:setattr(parent_node,'_ftact_side_info',msg),
                    qos_profile_sensor_data,callback_group=ReentrantCallbackGroup())
            self.side_images=parent_node._ftact_side_images
        parent_node._ftact_instances = getattr(parent_node, "_ftact_instances", 0) + 1
        self.instance = parent_node._ftact_instances
        self.write({"type": "model_loaded", "checkpoint_step": payload["step"],
            "smoke_test_checkpoint": payload["smoke_test"], "state_joint_names": list(STATE_JOINT_NAMES),
            "action_frame": "base_link", "action_representation": "absolute_tcp_pose_xyzw",
            "chunk_size": payload["config"]["chunk_size"],
            "replan_every_steps": self.settings.get('n_action_steps', 10) if self.timed_chunks else 1,
            "execution_mode": ('act_temporal_ensemble' if self.temporal_ensemble else
                               'flow_receding_horizon' if self.flow_matching else
                               'act_receding_horizon' if self.act_receding else
                               'timed_chunk_replan' if self.timed_chunks else 'first_action'),
            "ode_steps": self.settings.get('ode_steps') if self.flow_matching else None,
            "ode_solver": self.settings.get('ode_solver') if self.flow_matching else None,
            "noise_seed": self.settings.get('noise_seed') if self.flow_matching else None,
            "temporal_ensemble_coeff": self.settings.get('temporal_ensemble_coeff') if self.temporal_ensemble else None,
            "temporal_ensemble_order": 'oldest_first' if self.temporal_ensemble else None,
            "checkpoint_format": payload['format'],
            "camera_order":list(CAMERA_KEYS) if self.multiview else None,
            "multiview":self.multiview,
            "pace_phase_event": self.pace,
            "pace_phase_alpha": payload['config'].get('phase_alpha') if self.pace else None,
            "pace_tactile_history": payload['config'].get('tactile_history') if self.pace else None,
            "pace_correction_horizon": payload['config'].get('correction_horizon') if self.pace else None,
            "ground_truth_used": False, "force_input": True, "device": self.device,
            "amp": self.settings["amp"], "dry_run": self.settings["dry_run"]})
        self.get_logger().info(f"FTACT loaded training step {payload['step']}; {len(self.camera_names)} RGB + state6 + wrench6, TCP pose output")

    def write(self, value):
        value["policy_instance"] = self.instance
        self.journal.write(json.dumps(value, allow_nan=False) + "\n")

    def active(self):
        goal = self._parent_node.goal_handle
        return (rclpy.ok() and self._parent_node.is_active and goal is not None
                and goal.is_active and not goal.is_cancel_requested)

    def motion(self, command):
        return MotionUpdate(
            header=Header(frame_id="base_link", stamp=self.time_now().to_msg()),
            pose=Pose(position=Point(x=float(command[0]), y=float(command[1]), z=float(command[2])),
                orientation=Quaternion(x=float(command[3]), y=float(command[4]), z=float(command[5]), w=float(command[6]))),
            target_stiffness=np.diag(self.settings["stiffness"]).astype(float).flatten().tolist(),
            target_damping=np.diag(self.settings["damping"]).astype(float).flatten().tolist(),
            feedforward_wrench_at_tip=Wrench(force=Vector3(), torque=Vector3()),
            wrench_feedback_gains_at_tip=[.5, .5, .5, 0., 0., 0.],
            trajectory_generation_mode=TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION))

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        # No queued actions, command history, or latent state survives a new task/scene.
        self.model.policy.reset()
        if self.multiview:
            self.side_images.clear()
            self.source_joints.clear()
        if self.pace_context is not None:
            self.pace_context.reset()
        if self.flow_matching:
            self.model.reset_sampler(self.settings['noise_seed'])
        if self.task_contract and (task.target_module_name != self.task_contract['target_module']
                or task.port_name != self.task_contract['port'] or task.plug_type != 'sfp'):
            raise ValueError('This model was trained only for SFP insertion into nic_card_mount_0/sfp_port_0')
        queue = TimedActionChunk(20, self.settings.get('n_action_steps', 10)) if self.timed_chunks else None
        ensemble = (ACTTemporalPoseEnsembler(20, 20, self.settings['temporal_ensemble_coeff'])
                    if self.temporal_ensemble else None)
        start = self.time_now().nanoseconds * 1e-9
        budget = min(float(task.time_limit), self.settings["max_task_seconds"])
        task_id = task.id
        self._parent_node._ftact_tasks = getattr(self._parent_node, "_ftact_tasks", 0) + 1
        task_label = f"{self._parent_node._ftact_tasks:02d}_" + re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)[:80]
        previous = None
        last_stamp = -float("inf")
        last_fresh_wall = time.monotonic()
        next_tick = start
        count = dropped = inferences = 0
        chunk_elapsed = 0.
        timestamp_logged = False
        inputs_saved = False
        skipped = {"missing_observation": 0, "old_or_duplicate_observation": 0, "initializing": 0}
        last_initialization_error = None
        self.write({"type": "task_start", "task_id": task_id, "simulation_time": start,
            "plug_type": task.plug_type, "port_name": task.port_name,
            "target_module_name": task.target_module_name, "budget_seconds": budget})
        reason, completed = "canceled", False
        try:
            while self.active():
                now = self.time_now().nanoseconds * 1e-9
                if now - start >= budget:
                    reason, completed = "control_window_completed", True
                    break
                if time.monotonic() - last_fresh_wall > self.settings["max_wall_stall_seconds"]:
                    raise TimeoutError("No fresh usable observation or simulation clock stalled")
                observation = get_observation()
                if observation is None or now < next_tick:
                    if observation is None:
                        skipped["missing_observation"] += 1
                    time.sleep(.003)
                    continue
                stamp = stamp_seconds(observation.center_image.header.stamp)
                if not timestamp_logged:
                    self.write({"type": "observation_timestamps", "task_id": task_id,
                        "now": now, "camera": stamp,
                        "joints": stamp_seconds(observation.joint_states.header.stamp),
                        "controller": stamp_seconds(observation.controller_state.header.stamp),
                        "wrench": stamp_seconds(observation.wrist_wrench.header.stamp)})
                    timestamp_logged = True
                if (stamp <= last_stamp or stamp < start-.1 or now-stamp > self.settings["max_observation_age_seconds"]):
                    skipped["old_or_duplicate_observation"] += 1
                    time.sleep(.003)
                    continue
                sync_details={}
                try:
                    if self.multiview:
                        arrays,joints,sync_details=multiview_arrays(observation,tuple(self.side_images),
                            tuple(self.source_joints),self._parent_node._ftact_side_info,
                            self.settings['collection_settings'],self.settings['expected_camera_info'])
                    else:
                        candidates = tuple(self.source_joints)
                        joints = next((j for j in reversed(candidates)
                            if stamp_seconds(j.header.stamp) <= stamp + .001), None)
                        if joints is None or stamp-stamp_seconds(joints.header.stamp) > .1:
                            raise ValueError("Waiting for time-aligned /joint_states including gripper")
                        arrays = observation_arrays(observation, self.settings["state_joint_names"], joints)
                    actual = unit_pose(pose_array(observation.controller_state.tcp_pose))
                except ValueError as error:
                    if (self.multiview and isinstance(error,ObservationNotReady)) or (count == 0 and now-start < 5):
                        skipped["initializing"] += 1
                        if str(error) != last_initialization_error:
                            self.write({"type": "waiting_for_initial_observation", "task_id": task_id, "reason": str(error)})
                            self.get_logger().info(f"Waiting for observation initialization: {error}")
                            last_initialization_error = str(error)
                        time.sleep(.01)
                        continue
                    raise
                last_stamp = stamp
                last_fresh_wall = time.monotonic()
                if observation.controller_state.header.frame_id not in ("", "base_link"):
                    raise ValueError("Controller TCP pose is not in base_link")
                wrench = arrays["observation.environment_state"]
                if np.linalg.norm(wrench[:3]) > self.settings["max_force_n"] or np.linalg.norm(wrench[3:]) > self.settings["max_torque_nm"]:
                    raise ValueError("Measured wrist wrench exceeded configured execution limits")
                if self.pace_context is not None:
                    arrays.update(self.pace_context.update(wrench, stamp))
                if not inputs_saved:
                    np.savez(self.output / f"task_{task_label}_first_observation.npz", **arrays)
                    for camera in self.camera_names:
                        cv2.imwrite(str(self.output / f"task_{task_label}_{camera}.png"),
                            arrays[f"observation.images.{camera}_camera"][..., ::-1])
                    self.write({"type": "observation_contract", "task_id": task_id,
                        "joint_names_received": list(joints.name),
                        "adapter_joint_names": list(observation.joint_states.name),
                        "source_joint_topic": "/joint_states",
                        "source_joint_stamp": stamp_seconds(joints.header.stamp),
                        "source_state": arrays["observation.state"].tolist(),
                        "wrench": wrench.tolist(), "wrench_frame": observation.wrist_wrench.header.frame_id,
                        "tcp_frame": observation.controller_state.header.frame_id,
                        "camera_shapes": {k: list(v.shape) for k,v in arrays.items() if "images" in k},
                        **sync_details})
                    inputs_saved = True
                now = self.time_now().nanoseconds * 1e-9
                new_chunk = queue is None or queue.needs_refill(now)
                if new_chunk:
                    batch = {}
                    for key, array in arrays.items():
                        tensor = torch.from_numpy(array.copy())
                        if key.startswith("observation.images."):
                            tensor = tensor.permute(2, 0, 1).float() / 255
                        batch[key] = tensor.unsqueeze(0).to(self.device)
                    began = time.monotonic()
                    precision = (torch.autocast("cuda", dtype=torch.bfloat16 if self.settings["amp"] == "bf16" else torch.float16)
                                 if self.settings["amp"] != "none" else nullcontext())
                    with torch.inference_mode(), precision:
                        chunk = self.model.predict_action_chunk(batch)[0].cpu().numpy()
                    chunk_elapsed = time.monotonic() - began
                    inferences += 1
                    self.write({'type': 'chunk_prediction', 'task_id': task_id,
                        'inference_id': inferences, 'observation_stamp': stamp,
                        'inference_seconds': chunk_elapsed, 'first_target': chunk[0].tolist(),
                        'last_target': chunk[-1].tolist(),
                        'actions': chunk.tolist() if self.flow_matching else None,
                        'control_period_exceeded': chunk_elapsed > 1/self.settings['rate_hz']})
                elapsed = chunk_elapsed if new_chunk else 0.
                now = self.time_now().nanoseconds * 1e-9
                if not self.active():
                    break
                if now - stamp > self.settings["max_observation_age_seconds"]:
                    dropped += 1
                    self.write({"type": "stale_prediction_dropped", "task_id": task_id,
                        "observation_age_seconds": now-stamp, "inference_seconds": elapsed})
                    continue
                if previous is None:
                    previous = actual
                ensemble_details = {}
                if ensemble is not None:
                    ensemble.add(chunk, stamp, inferences)
                    target, ensemble_details = ensemble.target(now)
                    chunk_index = ensemble_details['ensemble_source_indices'][-1]
                elif queue is not None:
                    if new_chunk:
                        queue.replace(chunk, stamp)
                    if queue.index(now) >= len(queue.chunk):
                        dropped += 1
                        continue
                    target, chunk_index = queue.target(now)
                else:
                    target, chunk_index = chunk[0], 0
                command, limited = bounded_pose(target, previous, self.minimum, self.maximum,
                    self.settings["max_translation_step_m"], self.settings["max_rotation_step_rad"])
                if not self.settings["dry_run"]:
                    accepted = move_robot(motion_update=self.motion(command))
                    if accepted is False:
                        raise RuntimeError("Official model interface rejected the TCP pose command")
                previous = command
                count += 1
                if ensemble is not None:
                    # Keep the grid anchored to the camera. Inference overruns
                    # skip expired ticks rather than shifting the whole horizon.
                    next_tick = ensemble_details['ensemble_target_time']
                elif queue is not None:
                    next_tick += (max(0, int(np.floor((now-next_tick)*self.settings['rate_hz']))) + 1) / self.settings['rate_hz']
                else:
                    next_tick = now + 1/self.settings["rate_hz"]
                self.write({"type": "prediction", "task_id": task_id, "step": count,
                    "simulation_time": now, "observation_stamp": stamp,
                    "inference_seconds": elapsed, "observation_age_seconds": now-stamp,
                    "actual_tcp_pose": actual.tolist(), "raw_target": target.tolist(),
                    "chunk_index": chunk_index, "inference_id": inferences,
                    "chunk_observation_stamp": queue.reference_time if queue is not None else stamp,
                    "new_chunk": new_chunk,
                    "command": command.tolist(), "command_sent": not self.settings["dry_run"],
                    "force_norm": float(np.linalg.norm(wrench[:3])), **ensemble_details, **sync_details, **limited})
                if count == 1 or count % 20 == 0:
                    send_feedback(f"FTACT: {count} predictions; {elapsed*1000:.1f} ms inference")
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            self.get_logger().error(f"FTACT stopped: {reason}")
            # End with a current-pose hold while this task still owns the controller.
            if self.active() and not self.settings["dry_run"]:
                try:
                    hold = unit_pose(pose_array(get_observation().controller_state.tcp_pose))
                    move_robot(motion_update=self.motion(hold))
                except Exception:
                    pass
        completed = completed and count > 0
        if reason == "control_window_completed" and not completed:
            reason = "no_usable_predictions"
        self.write({"type": "task_end", "task_id": task_id, "predictions": count, 'model_inferences': inferences,
            "stale_predictions_dropped": dropped, "reason": reason,
            "skipped": skipped,
            "last_initialization_error": last_initialization_error,
            "policy_execution_completed": completed, "insertion_success": "determined_by_official_scoring"})
        return completed
