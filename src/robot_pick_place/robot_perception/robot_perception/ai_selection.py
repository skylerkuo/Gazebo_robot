"""Associate a semantic image mask to an existing 3D collision object.

RGB and cloud must be pixel-registered. No 2D-only or shape-match fallback is used.
"""
import numpy as np
from sensor_msgs_py.point_cloud2 import dtype_from_fields


def quaternion_matrix(quaternion):
    q = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float)
    length = np.linalg.norm(q)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError('Invalid rotation quaternion')
    x, y, z, w = q / length
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def registered_mask_points(cloud, image, mask, max_stamp_delta=0.15):
    if cloud.height <= 1 or (cloud.height, cloud.width) != (image.height, image.width):
        raise ValueError('AI requires an organized PointCloud2 registered to RGB at the same resolution')
    if mask.shape != (image.height, image.width):
        raise ValueError('Segmentation mask does not match the original RGB dimensions')
    stamp = lambda header: header.stamp.sec + header.stamp.nanosec * 1e-9
    if abs(stamp(cloud.header) - stamp(image.header)) > max_stamp_delta:
        raise ValueError('RGB and point cloud timestamps are not synchronized')
    if not cloud.header.frame_id:
        raise ValueError('Point cloud has no coordinate frame')
    dtype = dtype_from_fields(cloud.fields, point_step=cloud.point_step)
    dtype = dtype.newbyteorder('>' if cloud.is_bigendian else '<')
    # Honor both row padding and point padding; preserve original pixel indices.
    data = np.ndarray((cloud.height, cloud.width), dtype=dtype, buffer=cloud.data,
                      strides=(cloud.row_step, cloud.point_step))
    xyz = np.stack([data[name] for name in ('x', 'y', 'z')], axis=-1)
    points = xyz[mask.astype(bool)]
    return points[np.isfinite(points).all(axis=1)]


def transform_points(points, transform):
    rotation = quaternion_matrix(transform.rotation)
    translation = transform.translation
    return points @ rotation.T + [translation.x, translation.y, translation.z]


def select_scene_object(points, objects, support_id, target_frame,
                        padding=0.01, min_fraction=0.5, ambiguity_margin=0.15):
    """Score visible semantic points against primitive volumes, excluding the support."""
    if len(points) < 20:
        raise ValueError('Not enough valid depth points inside the AI mask (need at least 20)')
    scores = []
    for obj in objects:
        if obj.id == support_id:
            continue
        if obj.header.frame_id != target_frame:
            raise ValueError(f'Collision object {obj.id!r} has an unexpected frame')
        if len(obj.primitives) != 1 or len(obj.primitive_poses) != 1:
            continue
        primitive, pose = obj.primitives[0], obj.primitive_poses[0]
        local = (points - [pose.position.x, pose.position.y, pose.position.z]) @ quaternion_matrix(pose.orientation)
        dims = np.asarray(primitive.dimensions)
        if primitive.type == primitive.BOX and len(dims) == 3:
            inside = (np.abs(local) <= dims / 2 + padding).all(axis=1)
        elif primitive.type == primitive.CYLINDER and len(dims) == 2:
            inside = ((local[:, :2]**2).sum(axis=1) <= (dims[1]+padding)**2) & (np.abs(local[:, 2]) <= dims[0]/2+padding)
        else:
            continue
        scores.append((float(inside.mean()), obj.id))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < min_fraction:
        raise ValueError(f'AI mask has no reliable 3D object association: {scores}')
    if len(scores) > 1 and scores[0][0] - scores[1][0] < ambiguity_margin:
        raise ValueError(f'AI mask overlaps multiple 3D objects ambiguously: {scores[:2]}')
    return scores[0]
