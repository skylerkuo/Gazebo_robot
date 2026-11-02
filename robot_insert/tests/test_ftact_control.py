"""Contracts at the learned policy / Gazebo boundary."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ftact_control import STATE_JOINT_NAMES, bounded_pose, observation_arrays, ros_image_rgb
import numpy as np


def image_message(encoding="rgb8"):
    # Red image with padding after each row. BGR changes the byte order, not the color.
    pixel = [0, 0, 255] if encoding == "bgr8" else [255, 0, 0]
    raw = np.array([pixel*2+[77,88], pixel*2+[99,11]], dtype=np.uint8)
    return NS(height=2,width=2,step=8,encoding=encoding,data=raw.tobytes())


class ControlTests(unittest.TestCase):
    def test_rgb_conversion_respects_row_padding_and_bgr_channel_order(self):
        a=ros_image_rgb(image_message())
        b=ros_image_rgb(image_message("bgr8"))
        self.assertEqual(a.shape,(256,288,3))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[0,0],[255,0,0])
        bad=image_message();bad.data=bad.data[:-1]
        with self.assertRaisesRegex(ValueError,"byte count"):
            ros_image_rgb(bad)

    def test_joint_names_reconstruct_source_order_and_wrench_is_mandatory(self):
        names=list(STATE_JOINT_NAMES)[::-1]+["wrist_3_joint"]
        positions=list(range(6,0,-1))+[42]
        observation=NS(joint_states=NS(name=names,position=positions),
            wrist_wrench=NS(wrench=NS(force=NS(x=1,y=2,z=3),torque=NS(x=4,y=5,z=6))),
            left_image=image_message(),center_image=image_message(),right_image=image_message())
        arrays=observation_arrays(observation)
        np.testing.assert_array_equal(arrays['observation.state'],np.arange(1,7))
        np.testing.assert_array_equal(arrays['observation.environment_state'],np.arange(1,7))
        raw_joints=observation.joint_states
        observation.joint_states=NS(name=['wrist_3_joint'],position=[42])
        arrays=observation_arrays(observation,source_joints=raw_joints)
        np.testing.assert_array_equal(arrays['observation.state'],np.arange(1,7))
        observation.joint_states=raw_joints
        observation.joint_states.name[0]='unknown'
        with self.assertRaisesRegex(ValueError,"Missing"):
            observation_arrays(observation)

    def test_command_limits_preserve_unit_quaternion_and_shortest_rotation(self):
        previous=np.array([0.,0,0,0,0,0,1])
        target=np.array([.1,0,0,0,0,np.sqrt(.5),np.sqrt(.5)])
        result,flags=bounded_pose(target,previous,np.full(3,-1.),np.ones(3),.005,.05)
        self.assertAlmostEqual(np.linalg.norm(result[:3]),.005)
        self.assertAlmostEqual(np.linalg.norm(result[3:]),1.)
        self.assertAlmostEqual(2*np.arccos(result[-1]),.05)
        self.assertTrue(flags['rotation_limited'] and flags['translation_limited'])
        target=previous.copy();target[3:]*=-1
        same,_=bounded_pose(target,previous,np.full(3,-1.),np.ones(3),.005,.05)
        np.testing.assert_allclose(same,previous)

    def test_invalid_or_outside_workspace_predictions_are_rejected(self):
        previous=np.array([0.,0,0,0,0,0,1])
        target=previous.copy();target[0]=2
        with self.assertRaisesRegex(ValueError,"workspace"):
            bounded_pose(target,previous,np.full(3,-1.),np.ones(3),.005,.05)
        with self.assertRaisesRegex(ValueError,"Invalid"):
            bounded_pose(np.zeros(7),previous,np.full(3,-1.),np.ones(3),.005,.05)


if __name__=='__main__':
    unittest.main()
