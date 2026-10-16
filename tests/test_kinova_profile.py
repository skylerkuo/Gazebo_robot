"""Integration contracts for the official six-axis Lite model and its 2F gripper."""
import tempfile
import subprocess
import unittest
from xml.etree import ElementTree as ET

import yaml
from robot_bringup.configuration import prepare_configuration
from robot_bringup.profile import load_profile


class KinovaProfileContract(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile('kinova_gen3_lite')
        self.runtime = tempfile.TemporaryDirectory()
        self.addCleanup(self.runtime.cleanup)

    def prepare(self, source='known'):
        return prepare_configuration(self.profile, self.runtime.name, source, source == 'perception')

    def test_six_axis_model_and_independent_gripper(self):
        config, task, _, urdf, control = self.prepare()
        robot = ET.parse(urdf).getroot()
        systems = robot.findall('ros2_control')
        self.assertEqual(len(systems), 2)
        self.assertTrue(all(s.findtext('hardware/plugin') == 'gz_ros2_control/GazeboSimSystem' for s in systems))
        names = [j.attrib['name'] for s in systems for j in s.findall('joint')]
        self.assertEqual(len(names), len(set(names)), 'A joint cannot be owned by both hardware components')
        arm = [j.attrib['name'] for j in systems[0].findall('joint')]
        self.assertEqual(arm, [f'joint_{i}' for i in range(1, 7)])
        self.assertNotIn('KortexMultiInterfaceHardware', config['robot_description'])
        self.assertEqual(robot.find('.//plugin/parameters').text, str(control))
        controllers = yaml.safe_load(control.read_text())
        self.assertEqual(controllers[task['controller_names'][0]]['ros__parameters']['joints'], arm)
        self.assertEqual(controllers[task['controller_names'][1]]['ros__parameters']['joint'], 'right_finger_bottom_joint')

    def test_mimic_limits_and_named_poses(self):
        config, task, _, urdf, _ = self.prepare()
        robot = ET.parse(urdf).getroot()
        master = robot.find("joint[@name='right_finger_bottom_joint']")
        for name, multiplier, offset in [('left_finger_bottom_joint', 1.0, 0.0),
                                          ('left_finger_tip_joint', -0.276, -0.1),
                                          ('right_finger_tip_joint', -0.276, -0.1)]:
            mimic = robot.find(f"joint[@name='{name}']/mimic")
            self.assertEqual(mimic.attrib['joint'], master.attrib['name'])
            self.assertAlmostEqual(float(mimic.attrib['multiplier']), multiplier)
            self.assertAlmostEqual(float(mimic.attrib['offset']), offset)
        srdf = ET.fromstring(config['robot_description_semantic'])
        self.assertEqual(srdf.find('end_effector').attrib['group'], task['gripper_group_name'])
        for parameter in ['gripper_open_pose', 'gripper_close_pose']:
            value = float(srdf.find(f"group_state[@name='{task[parameter]}']/joint").attrib['value'])
            self.assertGreaterEqual(value, float(master.find('limit').attrib['lower']))
            self.assertLessEqual(value, float(master.find('limit').attrib['upper']))

    def test_external_camera_and_task_frames(self):
        _, task, perception, urdf, _ = self.prepare('perception')
        robot = ET.parse(urdf).getroot()
        self.assertIsNotNone(robot.find("link[@name='scene_camera_link']"))
        self.assertEqual(robot.find('.//sensor').attrib['type'], 'rgbd_camera')
        # Empty fixed links can disappear during URDF -> SDF conversion, taking sensors with them.
        converted = subprocess.run(['gz', 'sdf', '-p', str(urdf)], capture_output=True, text=True, check=True, timeout=10)
        self.assertIsNotNone(ET.fromstring(converted.stdout).find('.//sensor'))
        self.assertEqual(perception['target_frame'], task['world_frame'])
        self.assertIsNotNone(robot.find(f"link[@name='{task['gripper_frame']}']"))
        self.assertEqual(task['object_dimensions'], [0.35, 0.02])

    def test_execution_group_covers_arm_and_gripper_trajectory(self):
        config, _, _, urdf, _ = self.prepare()
        robot = ET.parse(urdf).getroot()
        joints = {j.attrib['name']: j for j in robot.findall('joint')}
        srdf = ET.fromstring(config['robot_description_semantic'])
        groups = {g.attrib['name']: g for g in srdf.findall('group')}

        def actuated(name):
            group = groups[name]
            result = {j.attrib['name'] for j in group.findall('joint')
                      if joints[j.attrib['name']].attrib['type'] != 'fixed'
                      and joints[j.attrib['name']].find('mimic') is None}
            for subgroup in group.findall('group'):
                result.update(actuated(subgroup.attrib['name']))
            return result

        required = {f'joint_{i}' for i in range(1, 7)} | {'right_finger_bottom_joint'}
        self.assertTrue(any(actuated(name) == required for name in groups),
                        'MTC executor needs a JointModelGroup matching the merged seven-joint trajectory')


if __name__ == '__main__':
    unittest.main()
