"""Causal four-camera inference inputs matching the collected ACT dataset."""
import numpy as np

from ftact_control import observation_arrays,ros_image_rgb,stamp_seconds
from sideview_data import CAMERAS,latest_causal

CAMERA_KEYS=tuple(f'observation.images.{name}_camera' for name in CAMERAS)


class ObservationNotReady(ValueError):
    """A synchronized sensor sample has not arrived yet; retry this observation."""


def multiview_arrays(observation,side_images,joint_messages,side_info,profile,expected_info):
    stamp=stamp_seconds(observation.center_image.header.stamp)
    tolerance=profile['camera_tolerance_s']
    side=latest_causal([(stamp_seconds(m.header.stamp),m) for m in side_images],stamp,tolerance)
    joints=latest_causal([(stamp_seconds(m.header.stamp),m) for m in joint_messages],
                         stamp,profile['state_tolerance_s'])
    if side is None or side_info is None: raise ObservationNotReady('Waiting for aligned side-camera image/info')
    if joints is None: raise ObservationNotReady('Waiting for causal /joint_states including gripper')
    images=[observation.left_image,observation.center_image,observation.right_image,side[1]]
    infos=[observation.left_camera_info,observation.center_camera_info,observation.right_camera_info,side_info]
    camera_times={name:stamp_seconds(image.header.stamp) for name,image in zip(CAMERAS,images)}
    ages=np.array([stamp-value for value in camera_times.values()])
    if not np.isfinite(ages).all() or ages.min() < -1e-8 or ages.max()>tolerance+1e-8:
        raise ObservationNotReady('Waiting for causal synchronized wrist/side images')
    wrench_time=stamp_seconds(observation.wrist_wrench.header.stamp)
    if not -1e-8<=stamp-wrench_time<=profile['state_tolerance_s']+1e-8:
        raise ObservationNotReady('Waiting for causal wrist wrench')
    for name,image,info in zip(CAMERAS,images,infos):
        expected=expected_info[name]
        if (image.width,image.height)!=(profile['record_width'],profile['record_height']):
            raise ValueError(f'{name} image resolution differs from collection')
        if ((info.width,info.height)!=(expected['width'],expected['height']) or
                not np.allclose(info.k,expected['k'],rtol=1e-6,atol=1e-5)):
            raise ValueError(f'{name} camera calibration differs from collection')
    arrays=observation_arrays(observation,source_joints=joints[1])
    arrays[CAMERA_KEYS[-1]]=ros_image_rgb(side[1])
    if arrays[CAMERA_KEYS[-1]].std()<1: raise ObservationNotReady('Side camera is still blank')
    return arrays,joints[1],{
        'camera_order':list(CAMERA_KEYS),'camera_stamps':camera_times,
        'camera_max_age_seconds':float(ages.max()),'source_joint_stamp':joints[0],
        'joint_age_seconds':stamp-joints[0],'wrench_stamp':wrench_time,
        'wrench_age_seconds':stamp-wrench_time}
