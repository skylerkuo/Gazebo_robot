"""Offline contracts for profiles, resource flow, and preservation of legacy algorithms/stages."""
import copy
from pathlib import Path
import re
import os
from unittest.mock import patch
import tempfile
import unittest
from xml.etree import ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory
from robot_bringup.profile import ProfileError, load_profile, task_parameters
from robot_bringup.configuration import prepare_configuration

REPO = Path(__file__).resolve().parents[1]


class ProfileContract(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile('mycobot_280')

    def external(self, change):
        profile = copy.deepcopy(self.profile)
        change(profile)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.yaml'
            path.write_text(yaml.safe_dump(profile))
            return load_profile('mycobot_280', str(path))

    def test_unknown_robot_rejected(self):
        with self.assertRaisesRegex(ProfileError, 'does not exist'):
            load_profile('missing_robot')

    def test_missing_package_rejected(self):
        with self.assertRaisesRegex(ProfileError, 'not installed/sourced'):
            self.external(lambda p: p['moveit'].update(package='nonexistent_moveit_package'))

    def test_invalid_group_rejected(self):
        with self.assertRaisesRegex(ProfileError, 'planning.arm_group'):
            self.external(lambda p: p['planning'].update(arm_group='missing_arm'))

    def test_invalid_controller_rejected(self):
        with self.assertRaisesRegex(ProfileError, 'execution.gripper_controller'):
            self.external(lambda p: p['execution'].update(gripper_controller='missing_gripper'))

    def test_invalid_resource_rejected(self):
        with self.assertRaisesRegex(ProfileError, 'package-relative'):
            self.external(lambda p: p['robot_description'].update(xacro_file='/fixed/workspace/robot.xacro'))

    def test_invalid_frame_rejected(self):
        self.profile['planning']['ik_frame'] = 'missing_link'
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ProfileError, 'planning.ik_frame'):
            prepare_configuration(self.profile, directory, 'known', False)

    def test_shared_description_and_runtime_controllers(self):
        with tempfile.TemporaryDirectory() as directory:
            moveit, task, perception, urdf, controllers = prepare_configuration(self.profile, directory, 'known', False)
            self.assertEqual(moveit['robot_description'], urdf.read_text())
            robot = ET.fromstring(urdf.read_text())
            self.assertEqual(robot.find('.//plugin/parameters').text, str(controllers))
            data = yaml.safe_load(controllers.read_text())
            vendor = Path(get_package_share_directory('mycobot_moveit_config')) / self.profile['moveit']['files']['ros2_controllers']
            self.assertEqual(data, yaml.safe_load(vendor.read_text()))
            self.assertNotIn('${', controllers.read_text())
            self.assertEqual(task['end_effector'], self.profile['planning']['end_effector'])
            self.assertEqual(task['controller_names'], ['arm_controller', 'gripper_action_controller'])
            self.assertEqual(perception['target_frame'], 'base_link')
            self.assertEqual(task['object_pose'][2], 0.175)
            self.assertEqual(task['object_dimensions'], [0.35, 0.015])
            self.assertEqual(robot.findall('.//sensor'), [])

    def test_alternate_install_prefix(self):
        packages = ('robot_bringup', 'robot_task_core', 'mycobot_description', 'mycobot_moveit_config', 'mycobot_gazebo')
        with tempfile.TemporaryDirectory(prefix='different_prefix_') as directory:
            prefix = Path(directory)
            index = prefix / 'share/ament_index/resource_index/packages'
            index.mkdir(parents=True)
            for package in packages:
                (index / package).touch()
                (prefix / 'share' / package).symlink_to(get_package_share_directory(package), target_is_directory=True)
            with patch.dict(os.environ, AMENT_PREFIX_PATH=str(prefix) + os.pathsep + os.environ['AMENT_PREFIX_PATH']):
                profile = load_profile('mycobot_280')
                runtime = prefix / 'runtime'
                runtime.mkdir()
                moveit, _, _, _, controllers = prepare_configuration(profile, runtime, 'known', False)
                self.assertIn(str(prefix / 'share/mycobot_description'), moveit['robot_description'])
                self.assertEqual(ET.fromstring(moveit['robot_description']).find('.//plugin/parameters').text, str(controllers))

    def test_perception_enables_sensor_and_preserves_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            moveit, task, perception, _, _ = prepare_configuration(self.profile, directory, 'perception', True)
            self.assertTrue(ET.fromstring(moveit['robot_description']).findall('.//sensor'))
            self.assertEqual(task['object_dimensions'], [0.35, 0.0125])
            self.assertEqual(perception['crop_min_x'], 0.1)
            self.assertEqual(perception['service_name'], '/perception/get_planning_scene')

    def test_mtc_stage_order_preserved(self):
        old = (REPO / 'src/robot_pick_place/mycobot_ros2/mycobot_mtc_pick_place_demo/src/mtc_node.cpp').read_text()
        new = (REPO / 'src/robot_pick_place/robot_task_core/src/mtc_node.cpp').read_text()
        stages = lambda text: re.findall(r'make_unique<mtc::stages::(\w+)>\(\s*"([^"]+)"', text)
        self.assertGreater(len(stages(old)), 20)
        self.assertEqual(stages(old), stages(new))

    def test_pcl_helpers_preserved(self):
        for name in ('cluster_extraction', 'object_segmentation', 'plane_segmentation', 'normals_curvature_and_rsd_estimation'):
            old = (REPO / f'src/robot_pick_place/mycobot_ros2/mycobot_mtc_pick_place_demo/src/{name}.cpp').read_text()
            new = (REPO / f'src/robot_pick_place/robot_perception/src/{name}.cpp').read_text()
            self.assertEqual(old.replace('mycobot_mtc_pick_place_demo/', 'robot_perception/'), new)

    def test_no_vendor_dependency_in_generic_packages(self):
        for name in ('robot_task_core', 'robot_perception', 'robot_task_interfaces'):
            for path in (REPO / 'src' / 'robot_pick_place' / name).rglob('*'):
                if path.is_file() and path.suffix in ('.cpp', '.h', '.xml', '.py', '.txt', '.yaml'):
                    self.assertNotIn('mycobot_', path.read_text(), str(path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
