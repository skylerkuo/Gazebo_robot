"""Run a trained RGB ACT policy against the Kinova Gazebo controllers.

Use the existing PyTorch venv: python -m robot_bringup.act_rollout --help.
The dedicated act_gazebo.launch.py starts the matching scene without MTC.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time
import uuid

import numpy as np

from robot_bringup.act_execution import (JointLimits, ObservationBuffer, TemporalEnsembler,
                                         next_target_ns)
from robot_bringup.demonstration_images import CAMERAS, JOINT_NAMES


ARM = '/joint_trajectory_controller'
GRIPPER = '/gen3_lite_2f_gripper_controller/gripper_cmd'


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def create_runtime(policy, args, log):
    import rclpy
    from action_msgs.msg import GoalStatus
    from builtin_interfaces.msg import Duration
    from control_msgs.action import GripperCommand
    from control_msgs.msg import JointTrajectoryControllerState
    from controller_manager_msgs.srv import ListControllers
    from cv_bridge import CvBridge
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class Runtime(Node):
        def __init__(self):
            super().__init__('act_rollout', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.policy, self.options = policy, args
            self.log = log
            self.dt_ns = int(round(policy.control_dt * 1e9))
            if self.dt_ns != 100_000_000 or not set(policy.config.cameras) <= set(CAMERAS):
                raise ValueError('Live Kinova camera setup requires a 10 Hz checkpoint with known camera names')
            self.observations = ObservationBuffer(policy.config.cameras)
            self.ensemble = TemporalEnsembler(self.dt_ns, policy.config.chunk_size,
                                               args.ensemble_decay, not args.no_temporal_ensemble)
            self.clock_ns = self.controller_ns = 0
            self.clock_wall = time.monotonic()
            self.limits = None
            self.fatal = None
            self.active = set()
            self.query = None
            self.query_wall = self.next_query = 0.
            self.commanded = False
            self.stopping = False
            self.gripper_pending = None
            self.gripper_handle = None
            self.gripper_position = None
            self.gripper_serial = 0
            self.arm_target = None
            self.command_count = self.prediction_count = 0
            self.bridge = CvBridge()
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='act_inference')
            self.prediction = None
            self.anchor = self.last_observation = self.last_target = 0
            self.prediction_wall = 0.
            self.publisher = self.create_publisher(JointTrajectory, ARM + '/joint_trajectory', 1)
            self.gripper = ActionClient(self, GripperCommand, GRIPPER)
            self.manager = self.create_client(ListControllers, '/controller_manager/list_controllers')
            self.subscriptions_owned = [
                self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data),
                self.create_subscription(JointState, '/joint_states', self.on_state, qos_profile_sensor_data),
                self.create_subscription(JointTrajectoryControllerState, ARM + '/controller_state',
                                         self.on_controller, qos_profile_sensor_data),
                self.create_subscription(String, '/robot_description', self.on_description,
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            ]
            for name in policy.config.cameras:
                self.subscriptions_owned.append(self.create_subscription(Image, CAMERAS[name] + '/color/image_raw',
                    lambda msg, name=name: self.observations.add_image(name, stamp_ns(msg.header.stamp), msg),
                    QoSProfile(depth=5)))

        def event(self, kind, **values):
            self.log.write(json.dumps(dict(event=kind, sim_ns=self.clock_ns, **values), allow_nan=False) + '\n')
            self.log.flush()

        def on_clock(self, msg):
            now = stamp_ns(msg.clock)
            if self.clock_ns and now < self.clock_ns:
                self.fatal = 'Simulation clock reset; restart the rollout after resetting Gazebo'
            if now > self.clock_ns:
                self.clock_wall = time.monotonic()
            self.clock_ns = now

        def on_state(self, msg):
            self.observations.add_state(stamp_ns(msg.header.stamp), msg.name, msg.position)

        def on_controller(self, msg):
            self.controller_ns = stamp_ns(msg.header.stamp)

        def on_description(self, msg):
            if self.limits is None:
                try:
                    self.limits = JointLimits.from_urdf(msg.data)
                except (ValueError, TypeError, AttributeError) as error:
                    self.fatal = str(error)

        def poll_controllers(self):
            now = time.monotonic()
            if self.query is not None and self.query.done():
                result = self.query.result()
                self.active = {c.name for c in result.controller if c.state == 'active'} if result else set()
                self.query = None
            if self.query is not None and now - self.query_wall > 3.:
                self.manager.remove_pending_request(self.query)
                self.query.cancel()
                self.query = None
                self.active.clear()
            if self.query is None and now >= self.next_query and self.manager.service_is_ready():
                self.query = self.manager.call_async(ListControllers.Request())
                self.query_wall, self.next_query = now, now + 1.

        def missing(self):
            missing = []
            if not self.clock_ns:
                missing.append('/clock')
            if self.limits is None:
                missing.append('Gazebo robot_description')
            if not self.controller_ns:
                missing.append('arm controller state')
            if not {'joint_trajectory_controller', 'gen3_lite_2f_gripper_controller'} <= self.active:
                missing.append('active arm/gripper controllers')
            if not self.gripper.server_is_ready() or not self.publisher.get_subscription_count():
                missing.append('controller command interfaces')
            if self.observations.latest is None:
                missing.append('synchronized RGB and joint states')
            elif self.stale_streams():
                missing.append('fresh streams: ' + ', '.join(self.stale_streams()))
            return missing

        def stale_streams(self):
            streams = {
                'RGB': (self.observations.latest.stamp_ns, 350_000_000),
                'joint states': (self.observations.states[-1][0], 100_000_000),
                'controller feedback': (self.controller_ns, 350_000_000),
            }
            return [f'{name} age={(self.clock_ns - stamp) / 1e9:.3f}s'
                    for name, (stamp, limit) in streams.items() if self.clock_ns - stamp > limit]

        def infer(self, observation):
            started = time.monotonic()
            images = {name: self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                      for name, msg in observation.images.items()}
            actions = self.policy.predict(images, observation.qpos)
            return observation.stamp_ns, actions, time.monotonic() - started

        def update_prediction(self):
            if self.prediction is not None and self.prediction.done():
                stamp, actions, seconds = self.prediction.result()
                self.ensemble.add(stamp, actions)
                self.prediction_count += 1
                self.event('prediction', observation_ns=stamp, inference_seconds=seconds,
                           latency_sim_seconds=(self.clock_ns - stamp) / 1e9)
                self.prediction = None
            observation = self.observations.latest
            if self.prediction is None and observation.stamp_ns > self.last_observation:
                self.last_observation = observation.stamp_ns
                self.prediction_wall = time.monotonic()
                self.prediction = self.pool.submit(self.infer, observation)
            if self.prediction is not None and time.monotonic() - self.prediction_wall > args.inference_timeout:
                raise RuntimeError('ACT inference timed out')

        def arm_message(self, qpos, seconds):
            message = JointTrajectory()
            message.joint_names = JOINT_NAMES[:6]
            point = JointTrajectoryPoint()
            point.positions = [float(x) for x in qpos[:6]]
            ns = max(1, int(round(seconds * 1e9)))
            point.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            message.points = [point]
            # Zero stamp asks the controller to start on receipt. All six joints
            # are included; it linearly interpolates this one position waypoint.
            return message

        def send_gripper(self, position):
            if self.gripper_pending is not None and not self.gripper_pending.done():
                return
            if self.gripper_position is not None and abs(position - self.gripper_position) < .005:
                return
            goal = GripperCommand.Goal()
            # This controller forwards the independent revolute joint position
            # directly, in rad, despite the generic action's gap-size wording.
            goal.command.position = float(position)
            goal.command.max_effort = 10.
            self.gripper_position = float(position)
            self.gripper_serial += 1
            serial = self.gripper_serial
            self.gripper_pending = self.gripper.send_goal_async(goal)

            def accepted(future):
                handle = future.result()
                if not handle.accepted:
                    self.fatal = 'Gripper controller rejected the command'
                    return
                self.gripper_handle = handle
                if self.stopping:
                    handle.cancel_goal_async()
                    return

                def completed(result):
                    status = result.result().status
                    if serial == self.gripper_serial and status == GoalStatus.STATUS_ABORTED:
                        self.fatal = 'Gripper controller aborted the current command'
                handle.get_result_async().add_done_callback(completed)
            self.gripper_pending.add_done_callback(accepted)

        def step(self):
            if self.fatal:
                raise RuntimeError(self.fatal)
            now = self.clock_ns
            if time.monotonic() - self.clock_wall > args.sensor_timeout:
                raise RuntimeError('Simulation clock stopped advancing')
            stale = self.stale_streams()
            if stale:
                raise RuntimeError('Stale sensor streams: ' + ', '.join(stale))
            if not {'joint_trajectory_controller', 'gen3_lite_2f_gripper_controller'} <= self.active:
                raise RuntimeError('A required controller is no longer active')
            self.update_prediction()
            target_ns = next_target_ns(now, self.anchor, self.dt_ns)
            if target_ns <= self.last_target:
                return
            action = self.ensemble.at(target_ns)
            if action is None:
                if self.commanded:
                    raise RuntimeError('No unexpired ACT prediction for the next control time')
                return  # Initial GPU warmup; no movement until a valid chunk exists.
            measured = self.observations.states[-1][1]
            if not args.dry_run and self.arm_target is not None and np.max(np.abs(measured[:6] - self.arm_target)) > .35:
                raise RuntimeError('Arm tracking error exceeded 0.35 rad')
            seconds = (target_ns - now) / 1e9
            bounded = self.limits.bound(action, measured, seconds, args.max_joint_velocity, args.max_gripper_velocity)
            if not args.dry_run:
                self.publisher.publish(self.arm_message(bounded, seconds))
                self.send_gripper(bounded[6])
            self.commanded = True
            self.arm_target = bounded[:6].copy()
            self.last_target = target_ns
            self.command_count += 1
            self.event('dry_run' if args.dry_run else 'command', target_ns=target_ns,
                       duration_seconds=seconds, measured_qpos=measured.tolist(),
                       predicted_qpos=action.tolist(), bounded_qpos=bounded.tolist())

        def stop(self):
            self.stopping = True
            if self.commanded and not args.dry_run and self.observations.states and rclpy.ok():
                measured = self.observations.states[-1][1]
                self.publisher.publish(self.arm_message(measured, .1))
                if self.gripper_handle is not None:
                    self.gripper_handle.cancel_goal_async()  # Controller holds its current position.
                # Service any pending goal response and its cancellation before destroying the node.
                until = time.monotonic() + .4
                while rclpy.ok() and time.monotonic() < until:
                    rclpy.spin_once(self, timeout_sec=.01)
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.event('stopped', commands=self.command_count, predictions=self.prediction_count)

    return Runtime()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path, help='An explicit best.pt or last.pt file')
    parser.add_argument('--device', default='cuda:0', help='cpu, auto, cuda or cuda:N')
    parser.add_argument('--precision', choices=('auto', 'fp32', 'bf16', 'fp16'), default='auto',
                        help='auto follows checkpoint AMP setting and CUDA support; CPU uses fp32')
    parser.add_argument('--check-model', action='store_true',
                        help='Load/warm up checkpoint and print settings; exits without ROS or controller commands')
    parser.add_argument('--duration', type=float, default=45., help='Rollout length in simulation seconds')
    parser.add_argument('--ready-timeout', type=float, default=120., help='Startup wait in wall seconds')
    parser.add_argument('--sensor-timeout', type=float, default=3., help='Clock watchdog in wall seconds')
    parser.add_argument('--inference-timeout', type=float, default=10., help='Single inference timeout in wall seconds')
    parser.add_argument('--max-joint-velocity', type=float, default=.6, help='Arm position-command rate bound, rad/s')
    parser.add_argument('--max-gripper-velocity', type=float, default=.6, help='Gripper command bound, rad/s')
    parser.add_argument('--ensemble-decay', type=float, default=.01)
    parser.add_argument('--no-temporal-ensemble', action='store_true', help='Use newest time-aligned prediction only')
    parser.add_argument('--dry-run', action='store_true', help='Compute and log actions without sending any commands')
    parser.add_argument('--torch-threads', type=int, default=2)
    parser.add_argument('--log', type=Path, help='JSONL log; default runs/act_rollout/<unique>.jsonl')
    args = parser.parse_args(argv)
    for name in ('duration', 'ready_timeout', 'sensor_timeout', 'inference_timeout',
                 'max_joint_velocity', 'max_gripper_velocity'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name.replace('_', '-') + ' must be positive and finite')
    if args.torch_threads < 1 or not np.isfinite(args.ensemble_decay) or args.ensemble_decay < 0:
        parser.error('Invalid torch-threads or ensemble-decay')
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error('Checkpoint file does not exist: ' + str(args.checkpoint))

    import torch
    from robot_bringup.act_policy import ACTPolicy

    torch.set_num_threads(args.torch_threads)
    print(f'Loading {args.checkpoint} on {args.device}', flush=True)
    policy = ACTPolicy(args.checkpoint, device=args.device, precision=args.precision)
    device = policy.device
    print('Warming up the ACT policy; predicted warmup actions are discarded.', flush=True)
    warmup_seconds = policy.warmup()
    summary = policy.describe()
    if args.check_model:
        print(json.dumps(dict(**summary, warmup_seconds=warmup_seconds), indent=2), flush=True)
        return 0
    height, width = policy.image_size
    print(f'Model: format {summary["format_version"]}; {summary["model_parameters"]:,} parameters; '
          f'RGB {list(policy.config.cameras)} {width}x{height}; precision={policy.precision}; '
          f'z=0; {policy.config.chunk_size} actions at {policy.fps:g} Hz; checkpoint normalization', flush=True)

    import rclpy
    from rclpy.signals import SignalHandlerOptions
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    path = (args.log or Path('runs/act_rollout') / (identifier + '.jsonl')).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Gazebo ACT; dry_run={args.dry_run}; duration={args.duration:g} sim seconds; log={path}', flush=True)
    node = log = None
    result = 1

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    try:
        log = path.open('x')
        node = create_runtime(policy, args, log)
        node.event('configuration', checkpoint=str(args.checkpoint), device=str(device),
                   cameras=list(policy.config.cameras), chunk_size=policy.config.chunk_size,
                   control_dt=policy.control_dt, dry_run=args.dry_run,
                   temporal_ensemble=not args.no_temporal_ensemble, ensemble_decay=args.ensemble_decay,
                   duration_sim_seconds=args.duration, max_joint_velocity=args.max_joint_velocity,
                   max_gripper_velocity=args.max_gripper_velocity, precision=policy.precision,
                   checkpoint_settings=summary, warmup_seconds=warmup_seconds)
        deadline = time.monotonic() + args.ready_timeout
        next_report = 0.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            node.poll_controllers()
            if node.fatal:
                raise RuntimeError(node.fatal)
            missing = node.missing()
            if not missing:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Readiness timeout: ' + '; '.join(missing))
            if time.monotonic() >= next_report:
                print('Waiting for ' + '; '.join(missing), flush=True)
                next_report = time.monotonic() + 5.
        node.anchor = node.observations.latest.stamp_ns
        node.event('started', observation_ns=node.anchor)
        print('Ready; ACT is observing and ' + ('logging predictions.' if args.dry_run else 'controlling the robot.'), flush=True)
        next_report = time.monotonic() + 5.
        while rclpy.ok() and node.clock_ns - node.anchor < args.duration * 1e9:
            rclpy.spin_once(node, timeout_sec=.005)
            node.poll_controllers()
            node.step()
            if time.monotonic() >= next_report:
                print(f'  sim={(node.clock_ns - node.anchor) / 1e9:.1f}s; '
                      f'predictions={node.prediction_count}; commands={node.command_count}', flush=True)
                next_report = time.monotonic() + 5.
        if not node.command_count:
            raise RuntimeError('Rollout ended without any valid action predictions')
        result = 0
        node.event('duration_reached')
        print('Rollout duration reached; holding current position. Inspect the bottle placement in Gazebo.', flush=True)
    except KeyboardInterrupt:
        print('Interrupted; holding current position.', flush=True)
        if node is not None:
            node.event('interrupted')
        result = 130
    except Exception as error:
        print(f'ACT rollout stopped: {error}', flush=True)
        if node is not None:
            node.event('error', reason=str(error))
    finally:
        if node is not None:
            try:
                node.stop()
            finally:
                node.destroy_node()
        if log is not None:
            log.close()
        rclpy.try_shutdown()
        signal.signal(signal.SIGTERM, previous_term)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
