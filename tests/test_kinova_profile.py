"""Integration contracts for the official six-axis Lite model and its 2F gripper."""
import tempfile
import subprocess
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from robot_bringup.configuration import prepare_configuration
from robot_bringup.profile import load_profile


class KinovaProfileContract(unittest.TestCase):
    def test_collection_bin_matches_world_and_stays_outside_pick_crop(self):
        _, task, perception, _, _ = self.prepare('perception')
        self.assertTrue(task['drop_into_bin'])
        world = ET.parse(Path(__file__).resolve().parents[1] /
                         'src/robot_pick_place/kinova_gen3_lite_support/worlds/pick_place.sdf')
        names = {model.attrib['name'] for model in world.findall('.//model')}
        self.assertNotIn('apple', names)
        self.assertNotIn('target_cylinder', names)
        self.assertNotIn('small_carton', names)
        self.assertTrue({'plastic_bottle', 'tennis_ball', 'red_mug', task['object_name']} <= names)
        model = world.find(".//model[@name='collection_bin']")
        pose = list(map(float, model.findtext('pose').split()))
        self.assertEqual(pose, task['bin_pose'])
        collisions = model.findall('link/collision')
        self.assertEqual(len(collisions), 5)
        floor = model.find("link/collision[@name='floor']")
        dims = list(map(float, floor.findtext('geometry/box/size').split()))
        self.assertEqual(dims[:2], task['bin_dimensions'][:2])
        self.assertEqual(dims[2], task['bin_wall_thickness'])
        self.assertLess(pose[1] + task['bin_dimensions'][1] / 2, perception['crop_min_y'])
        self.assertGreater(task['place_pose'][2], pose[2] + task['bin_dimensions'][2])
        _, known, _, _, _ = self.prepare('known')
        for name in ('bin_pose', 'bin_dimensions', 'bin_wall_thickness', 'place_pose'):
            self.assertEqual(known[name], task[name])

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
        self.assertEqual(task['object_dimensions'], [0.1, 0.02])
        self.assertEqual(task['object_name'], 'drink_can')

    def test_three_collection_cameras_survive_sdf_and_wrist_follows_gripper(self):
        _, _, _, urdf, _ = prepare_configuration(
            self.profile, self.runtime.name, 'known', True, demonstration_cameras=True)
        robot = ET.parse(urdf).getroot()
        self.assertEqual(robot.find("joint[@name='scene_camera_joint']/origin").get('xyz'), '0.4 -0.6 0.9')
        wrist = robot.find("joint[@name='wrist_camera_joint']")
        self.assertEqual(wrist.find('parent').get('link'), 'gripper_base_link')
        converted = subprocess.run(['gz', 'sdf', '-p', str(urdf)], capture_output=True, text=True, check=True, timeout=10)
        sensors = ET.fromstring(converted.stdout).findall('.//sensor')
        self.assertEqual({s.findtext('topic') for s in sensors}, {'camera_head', 'camera_wrist/image', 'camera_side/image'})
        for sensor in sensors:
            self.assertEqual(sensor.findtext('update_rate'), '10')
            self.assertEqual(sensor.findtext('camera/image/width'), '640')
            self.assertEqual(sensor.findtext('camera/image/height'), '480')

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
