"""Semantic-to-3D association checks, independent of optional model weights."""
import unittest
import numpy as np
from geometry_msgs.msg import Pose, Transform
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import Image, PointCloud2, PointField
from shape_msgs.msg import SolidPrimitive

from robot_perception.ai_selection import registered_mask_points, select_scene_object, transform_points


def cylinder(name, x=0.0):
    obj = CollisionObject(id=name)
    obj.header.frame_id = 'base_link'
    pose = Pose()
    pose.orientation.w = 1.0
    pose.position.x = x
    obj.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[0.1, 0.02])]
    obj.primitive_poses = [pose]
    return obj


class AssociationTests(unittest.TestCase):
    def test_registered_pixels_preserve_padding_endian_and_filter_nan(self):
        for endian in ('<', '>'):
            with self.subTest(endian=endian):
                raw = bytearray(64)  # 2 rows, 2 xyz pixels each, 8 bytes of row padding
                xyz = np.ndarray((2, 2, 3), dtype=endian+'f4', buffer=raw, strides=(32, 12, 4))
                xyz[:] = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [float('nan'), 0, 0]]]
                cloud = PointCloud2(height=2, width=2, point_step=12, row_step=32,
                                    is_bigendian=endian == '>', data=bytes(raw))
                cloud.header.frame_id = 'camera'
                cloud.fields = [PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
                                for name, offset in [('x', 0), ('y', 4), ('z', 8)]]
                image = Image(height=2, width=2)
                points = registered_mask_points(cloud, image, np.array([[False, True], [True, True]]))
                np.testing.assert_allclose(points, [[4, 5, 6], [7, 8, 9]])
                image.header.stamp.sec = 1
                with self.assertRaisesRegex(ValueError, 'timestamps'):
                    registered_mask_points(cloud, image, np.ones((2, 2)))
                image.height = 1
                with self.assertRaisesRegex(ValueError, 'registered'):
                    registered_mask_points(cloud, image, np.ones((1, 2)))

    def test_transform_uses_sensor_frame_rotation_and_translation(self):
        transform = Transform()
        transform.rotation.z = np.sqrt(0.5)
        transform.rotation.w = np.sqrt(0.5)
        transform.translation.x = 0.3
        np.testing.assert_allclose(transform_points(np.array([[1, 0, 0]]), transform),
                                   [[0.3, 1, 0]], atol=1e-8)

    def test_select_semantic_target_preserves_support_and_other_objects(self):
        points = np.tile([0.3, 0, 0], (30, 1))
        objects = [cylinder('support', 0.3), cylinder('other', 0.6), cylinder('can', 0.3)]
        self.assertEqual(select_scene_object(points, objects, 'support', 'base_link'), (1.0, 'can'))
        self.assertEqual([obj.id for obj in objects], ['support', 'other', 'can'])

    def test_fail_closed_for_missing_depth_unmatched_and_ambiguous_objects(self):
        points = np.zeros((30, 3))
        with self.assertRaisesRegex(ValueError, 'at least 20'):
            select_scene_object(points[:2], [cylinder('can')], '', 'base_link')
        with self.assertRaisesRegex(ValueError, 'no reliable'):
            select_scene_object(points, [cylinder('far', 0.5)], '', 'base_link')
        with self.assertRaisesRegex(ValueError, 'ambiguously'):
            select_scene_object(points, [cylinder('a'), cylinder('b')], '', 'base_link')
        with self.assertRaisesRegex(ValueError, 'unexpected frame'):
            select_scene_object(points, [cylinder('a')], '', 'world')


if __name__ == '__main__':
    unittest.main()
