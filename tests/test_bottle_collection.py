"""Dataset integrity checks: world/label consistency and physical-lift filtering."""
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

import yaml

from robot_bringup.collect_bottle import TASK_TEXT, prepare_episode, verify_delivery


class BottleCollectionTest(unittest.TestCase):
    def test_only_bottle_changes_and_all_props_are_in_planning_scene(self):
        source = Path(__file__).resolve().parents[1] / (
            'src/robot_pick_place/kinova_gen3_lite_support/worlds/pick_place.sdf')
        pose = [0.345, 0.16, 0.06, 0.0, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'episode'
            metadata = prepare_episode(directory, source, pose)
            world = ET.parse(directory / 'world.sdf').getroot().find('world')
            original = ET.parse(source).getroot().find('world')
            self.assertEqual({model.get('name') for model in world.findall('model')},
                             {model.get('name') for model in original.findall('model')})
            for model in original.findall('model'):
                if model.get('name') != 'plastic_bottle':
                    self.assertEqual(ET.tostring(model), ET.tostring(
                        world.find(f"model[@name='{model.get('name')}']")))
            bottle = world.find("model[@name='plastic_bottle']")
            self.assertEqual(list(map(float, bottle.findtext('pose').split())), pose)
            parameters = yaml.safe_load((directory / 'task.yaml').read_text())['mtc_node']['ros__parameters']
            self.assertEqual(parameters['object_pose'], pose)
            self.assertEqual(parameters['object_dimensions'], [0.12, 0.018])
            self.assertFalse(parameters['pick_only'])
            self.assertTrue(parameters['drop_into_bin'])
            self.assertEqual(metadata['bin_pose_xyz_rpy'], [0.25, -0.52, 0.0, 0.0, 0.0, 0.0])
            self.assertEqual(parameters['bin_dimensions'], [0.32, 0.22, 0.1])
            from moveit_msgs.msg import PlanningSceneWorld
            from rclpy.serialization import deserialize_message
            obstacles = deserialize_message((directory / 'known_obstacles.cdr').read_bytes(), PlanningSceneWorld)
            self.assertEqual({o.id for o in obstacles.collision_objects},
                             {'orange', 'drink_can', 'red_mug', 'tennis_ball'})
            sphere = next(o for o in obstacles.collision_objects if o.id == 'tennis_ball')
            self.assertEqual(sphere.primitives[0].type, sphere.primitives[0].SPHERE)
            self.assertEqual(list(sphere.primitives[0].dimensions), [0.026])
            self.assertEqual(TASK_TEXT, 'pick up drink bottle')
            with self.assertRaises(FileExistsError):
                prepare_episode(directory, source, pose)

    def samples(self, lifted=True):
        return ([{'stamp_ns': 900_000_000, 'xyz': [0.35, 0.15, 0.06]}] +
                [{'stamp_ns': 1_200_000_000, 'xyz': [0.35, 0.15, 0.14 if lifted else 0.06]}] +
                [{'stamp_ns': 2_000_000_000 + i * 50_000_000,
                  'xyz': [0.25, -0.52, 0.07], 'orientation_xyzw': [0.0, 0.0, 0.0, 1.0]}
                 for i in range(21)])

    def check_delivery(self, samples):
        return verify_delivery(samples, 1_000_000_000, 2_000_000_000, 1.0, 0.03,
                               [0.35, 0.15, 0.06, 0.0, 0.0, 0.0],
                               [0.25, -0.52, 0.0, 0.0, 0.0, 0.0], [0.32, 0.22, 0.1], 0.01)

    def test_actual_lift_required_not_just_successful_trajectory(self):
        self.assertTrue(self.check_delivery(self.samples())['verified'])
        self.assertFalse(self.check_delivery(self.samples(lifted=False))['verified'])

    def test_bottle_outside_bin_is_rejected(self):
        samples = self.samples()
        samples[-1]['xyz'] = [0.35, 0.15, 0.06]
        self.assertFalse(self.check_delivery(samples)['verified'])

    def test_bottle_held_above_bin_is_rejected(self):
        samples = self.samples()
        for sample in samples[2:]:
            sample['xyz'][2] = 0.22
        self.assertFalse(self.check_delivery(samples)['verified'])

    def test_horizontal_bottle_inside_bin_is_accepted_but_wall_crossing_rejected(self):
        samples = self.samples()
        for sample in samples[2:]:
            sample['orientation_xyzw'] = [0.0, 2 ** -0.5, 0.0, 2 ** -0.5]
            sample['xyz'][2] = 0.028
        self.assertTrue(self.check_delivery(samples)['verified'])
        samples[-1]['xyz'][0] += 0.13
        self.assertFalse(self.check_delivery(samples)['verified'])

    def test_unknown_orientation_is_rejected(self):
        samples = self.samples()
        del samples[-1]['orientation_xyzw']
        self.assertFalse(self.check_delivery(samples)['verified'])

    def test_sparse_or_missing_pose_stream_is_rejected(self):
        self.assertFalse(self.check_delivery([])['verified'])
        samples = self.samples()
        self.assertFalse(self.check_delivery([samples[0], samples[1], samples[2], samples[-1]])['verified'])

    def test_initial_pose_mismatch_is_rejected(self):
        samples = self.samples()
        samples[0]['xyz'][0] = 0.50
        self.assertFalse(self.check_delivery(samples)['verified'])

    def test_unexpected_pose_frame_is_rejected(self):
        samples = self.samples()
        samples[0]['frame_id'] = 'bottle_link'
        self.assertFalse(self.check_delivery(samples)['verified'])


if __name__ == '__main__':
    unittest.main()
