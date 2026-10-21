"""Live ACT timing and ROS command contracts; no robot or training is started."""
from concurrent.futures import Future
import importlib.util
import io
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from robot_bringup.act_execution import (JointLimits, Observation, ObservationBuffer,
                                         TemporalEnsembler, next_target_ns)
from robot_bringup.demonstration_images import JOINT_NAMES


DT = 100_000_000
SECOND = 1_000_000_000


class AlignmentTest(unittest.TestCase):
    def test_all_views_wait_for_causal_state_and_canonicalize_joints(self):
        buffer = ObservationBuffer(('scene', 'wrist', 'side'))
        buffer.add_state(SECOND - 10_000_000, list(reversed(JOINT_NAMES)), list(reversed(range(7))))
        for name in buffer.pending:
            buffer.add_image(name, SECOND, name)
        self.assertIsNone(buffer.latest)
        buffer.add_state(SECOND + 10_000_000, JOINT_NAMES, [99.] * 7)
        np.testing.assert_array_equal(buffer.latest.qpos, range(7))
        self.assertEqual(buffer.latest.stamp_ns, SECOND)
        self.assertEqual(buffer.latest.images, dict(scene='scene', wrist='wrist', side='side'))

    def test_stale_state_and_misaligned_images_are_not_observations(self):
        buffer = ObservationBuffer(('scene', 'wrist'))
        buffer.add_state(SECOND - DT, JOINT_NAMES, [0.] * 7)
        buffer.add_image('scene', SECOND, 'too old')
        buffer.add_image('wrist', SECOND + DT, 'wrist')
        buffer.add_image('scene', SECOND + DT, 'scene')
        buffer.add_state(SECOND + DT + 1, JOINT_NAMES, [1.] * 7)
        self.assertIsNone(buffer.latest)  # Only causal state is 200 ms old.
        buffer.add_state(SECOND + 2 * DT, JOINT_NAMES, [2.] * 7)
        buffer.add_image('scene', SECOND + 2 * DT, 'new scene')
        buffer.add_image('wrist', SECOND + 2 * DT, 'new wrist')
        self.assertEqual(buffer.latest.images['scene'], 'new scene')

    def test_invalid_states_and_duplicate_frames_are_ignored(self):
        buffer = ObservationBuffer(('scene',))
        buffer.add_state(SECOND, JOINT_NAMES, [float('nan')] * 7)
        buffer.add_state(SECOND, JOINT_NAMES[:-1], [0.] * 6)
        self.assertFalse(buffer.states)
        buffer.add_state(SECOND, JOINT_NAMES, [0.] * 7)
        buffer.add_image('scene', SECOND, 'valid')
        buffer.add_image('scene', SECOND, 'duplicate')
        self.assertEqual(buffer.latest.images['scene'], 'valid')


class TemporalTest(unittest.TestCase):
    def test_future_offset_and_same_time_overlap_including_zero_predictions(self):
        ensemble = TemporalEnsembler(DT, 3, decay=0.)
        first = np.repeat(np.array([[0.], [2.], [3.]]), 7, axis=1)
        ensemble.add(SECOND, first)
        self.assertIsNone(ensemble.at(SECOND))
        np.testing.assert_array_equal(ensemble.at(SECOND + DT), [0.] * 7)
        ensemble.add(SECOND + DT, np.full((3, 7), 4.))
        np.testing.assert_array_equal(ensemble.at(SECOND + 2 * DT), [3.] * 7)

    def test_late_inference_selects_future_row_and_never_extrapolates(self):
        ensemble = TemporalEnsembler(DT, 3)
        ensemble.add(SECOND, np.repeat(np.arange(1, 4)[:, None], 7, axis=1))
        np.testing.assert_array_equal(ensemble.at(SECOND + 3 * DT), [3.] * 7)
        self.assertIsNone(ensemble.at(SECOND + 4 * DT))
        self.assertFalse(ensemble.chunks)

    def test_latest_mode_and_exponential_weight_order(self):
        ensemble = TemporalEnsembler(DT, 3, decay=np.log(2.))
        ensemble.add(SECOND, np.zeros((3, 7)))
        ensemble.add(SECOND + DT, np.full((3, 7), 3.))
        np.testing.assert_allclose(ensemble.at(SECOND + 2 * DT), [1.] * 7)
        ensemble.enabled = False
        np.testing.assert_allclose(ensemble.at(SECOND + 2 * DT), [3.] * 7)

    def test_invalid_or_reordered_chunks_are_rejected(self):
        ensemble = TemporalEnsembler(DT, 3)
        with self.assertRaises(ValueError):
            ensemble.add(SECOND, np.full((3, 7), np.inf))
        ensemble.add(SECOND, np.zeros((3, 7)))
        with self.assertRaises(ValueError):
            ensemble.add(SECOND, np.zeros((3, 7)))

    def test_grid_allows_transport_time_without_changing_observation_origin(self):
        self.assertEqual(next_target_ns(SECOND + 2 * DT + 20_000_000, SECOND, DT), SECOND + 3 * DT)
        self.assertEqual(next_target_ns(SECOND + 3 * DT - 1, SECOND, DT), SECOND + 4 * DT)


class LimitsTest(unittest.TestCase):
    def test_clips_position_and_rate_but_keeps_gripper_as_seventh_dimension(self):
        limits = JointLimits(np.array([-1.] * 6 + [0.]), np.ones(7), np.full(7, .5))
        action = limits.bound([9.] * 7, np.zeros(7), .1, .2, .3)
        np.testing.assert_allclose(action, [.02] * 6 + [.03])
        with self.assertRaises(ValueError):
            limits.bound([np.nan] * 7, np.zeros(7), .1)

    def test_hardware_description_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'GazeboSimSystem'):
            JointLimits.from_urdf('<robot><ros2_control><hardware><plugin>real_robot</plugin></hardware></ros2_control></robot>')

    @unittest.skipUnless(importlib.util.find_spec('ament_index_python'), 'ROS environment not sourced')
    def test_live_limits_are_read_from_generated_three_camera_urdf(self):
        import tempfile
        from robot_bringup.configuration import prepare_configuration
        from robot_bringup.profile import load_profile
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, urdf, _ = prepare_configuration(load_profile('kinova_gen3_lite'), directory,
                                                     'known', True, demonstration_cameras=True)
            limits = JointLimits.from_urdf(urdf.read_text())
        self.assertEqual(limits.lower.shape, (7,))
        self.assertLessEqual(limits.lower[6], 0.)
        self.assertGreater(limits.upper[6], .7)
        self.assertEqual(limits.velocity[6], .6)


@unittest.skipUnless(importlib.util.find_spec('rclpy'), 'ROS environment not sourced')
class ROSCommandTest(unittest.TestCase):
    def runtime(self, dry_run=False):
        from robot_bringup.act_rollout import create_runtime
        policy = SimpleNamespace(control_dt=.1, config=SimpleNamespace(cameras=('scene',), chunk_size=3))
        args = SimpleNamespace(dry_run=dry_run, ensemble_decay=.01, no_temporal_ensemble=False,
                               max_joint_velocity=.6, max_gripper_velocity=.6,
                               inference_timeout=10., sensor_timeout=3.)
        with patch('rclpy.node.Node.__init__', return_value=None), \
                patch('rclpy.node.Node.create_publisher', return_value=MagicMock()), \
                patch('rclpy.node.Node.create_subscription', return_value=MagicMock()), \
                patch('rclpy.node.Node.create_client', return_value=MagicMock()), \
                patch('rclpy.action.ActionClient', return_value=MagicMock()):
            node = create_runtime(policy, args, io.StringIO())
        node.anchor = node.clock_ns = node.controller_ns = SECOND
        node.clock_wall = time.monotonic()
        node.active = {'joint_trajectory_controller', 'gen3_lite_2f_gripper_controller'}
        node.limits = JointLimits(np.array([-2.] * 6 + [0.]), np.ones(7) * 2., np.ones(7))
        node.observations.states.append((SECOND, np.zeros(7)))
        node.observations.latest = Observation(SECOND, {}, np.zeros(7))
        node.last_observation = SECOND
        node.ensemble.add(SECOND, np.full((3, 7), .02))
        self.addCleanup(node.pool.shutdown, wait=True)
        return node

    def test_controller_command_has_six_arm_joints_and_independent_gripper_action(self):
        node = self.runtime()
        node.step()
        message = node.publisher.publish.call_args.args[0]
        self.assertEqual(message.joint_names, JOINT_NAMES[:6])
        np.testing.assert_allclose(message.points[0].positions, [.02] * 6)
        self.assertEqual(message.points[0].time_from_start.nanosec, DT)
        gripper_goal = node.gripper.send_goal_async.call_args.args[0]
        self.assertAlmostEqual(gripper_goal.command.position, .02)
        self.assertEqual(gripper_goal.command.max_effort, 10.)
        node.step()
        self.assertEqual(node.publisher.publish.call_count, 1)  # One command per time slot.

    def test_readiness_waits_until_initial_queued_images_are_fresh(self):
        node = self.runtime()
        self.assertFalse(node.missing())
        node.clock_ns += 5 * DT
        self.assertTrue(any('fresh streams' in reason for reason in node.missing()))

    def test_dry_run_never_sends_arm_or_gripper_commands_including_shutdown(self):
        node = self.runtime(dry_run=True)
        node.step()
        node.stop()
        node.publisher.publish.assert_not_called()
        node.gripper.send_goal_async.assert_not_called()
        self.assertEqual(node.command_count, 1)

    def test_sensor_staleness_clock_reset_and_expired_predictions_stop_execution(self):
        for reason in ('stale', 'reset', 'expired'):
            node = self.runtime()
            if reason == 'stale':
                node.clock_ns += 4 * DT
            elif reason == 'reset':
                node.fatal = 'Simulation clock reset'
            else:
                node.commanded = True
                node.ensemble.chunks.clear()
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                node.step()
            node.publisher.publish.assert_not_called()

    def test_asynchronous_prediction_uses_source_timestamp(self):
        node = self.runtime()
        node.ensemble.chunks.clear()
        node.prediction = Future()
        node.prediction.set_result((SECOND, np.ones((3, 7)), .04))
        node.update_prediction()
        self.assertEqual(node.ensemble.chunks[0][0], SECOND)
        event = json.loads(node.log.getvalue().splitlines()[-1])
        self.assertEqual(event['observation_ns'], SECOND)

    def test_stop_replaces_arm_trajectory_with_measured_hold_and_cancels_gripper(self):
        node = self.runtime()
        node.commanded = True
        node.gripper_handle = MagicMock()
        with patch('rclpy.ok', return_value=True), patch('rclpy.spin_once'), \
                patch('robot_bringup.act_rollout.time.monotonic', side_effect=[0., 1.]):
            node.stop()
        hold = node.publisher.publish.call_args.args[0]
        np.testing.assert_array_equal(hold.points[0].positions, np.zeros(6))
        node.gripper_handle.cancel_goal_async.assert_called_once()

    def test_clock_reset_callback_and_tracking_error_are_detected(self):
        from rosgraph_msgs.msg import Clock
        node = self.runtime()
        node.on_clock(Clock())
        with self.assertRaisesRegex(RuntimeError, 'clock reset'):
            node.step()
        node = self.runtime()
        node.arm_target = np.ones(6)
        with self.assertRaisesRegex(RuntimeError, 'tracking error'):
            node.step()


if __name__ == '__main__':
    unittest.main()
