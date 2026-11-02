"""Geometry, causal alignment, and streaming storage for four-view SFP demonstrations.

ROS-independent helpers. Camera poses in SDF use X forward / Z up; saved optical
poses use X right / Y down / Z forward. Recorded commands are absolute base_link
TCP targets, not measured TCP positions. Future-action chunks must be shifted by
one row and must not cross recording gaps or episode boundaries.
"""
import copy
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

import numpy as np

CAMERAS = ('left', 'center', 'right', 'side')
ARM_JOINTS = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
LEGACY_JOINTS = ('elbow_joint', 'gripper/left_finger_joint', 'shoulder_lift_joint',
                 'shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint')
TASK = {'plug': 'sfp_tip', 'target_module': 'nic_card_mount_0', 'port': 'sfp_port_0'}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_settings(settings):
    if settings['format'] != 'aic-sfp0-sideview-v1' or settings['fps'] != 20:
        raise ValueError('Expected the 20 Hz SFP0 side-view collection contract')
    for key in ('record_width', 'record_height', 'minimum_frames'):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f'Invalid {key}')
    for key in ('camera_tolerance_s', 'state_tolerance_s', 'maximum_frame_gap_s'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not 0 < settings['real_time_factor'] <= 1:
        raise ValueError('Collection real_time_factor must be in (0, 1]')
    camera = settings['camera']
    for key in ('position_world', 'look_at_world'):
        if np.asarray(camera[key]).shape != (3,) or not np.isfinite(camera[key]).all():
            raise ValueError('Camera pose must contain three finite coordinates')
    if not 0 < camera['near'] < camera['far'] or not 0 < camera['horizontal_fov'] < math.pi:
        raise ValueError('Invalid camera clipping planes or field of view')
    for value in settings['variation'].values():
        if not math.isfinite(value) or value < 0:
            raise ValueError('Scene variations must be finite, nonnegative')
    camera_geometry(camera)


def camera_geometry(camera):
    position = np.asarray(camera['position_world'], dtype=float)
    delta = np.asarray(camera['look_at_world'], dtype=float) - position
    if np.linalg.norm(delta) < 1e-6 or np.linalg.norm(delta[:2]) < 1e-6:
        raise ValueError('Choose a nonvertical, nonzero camera viewing direction')
    yaw = math.atan2(delta[1], delta[0])
    pitch = math.atan2(-delta[2], np.linalg.norm(delta[:2]))
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rotation = np.array([[cy*cp, -sy, cy*sp], [sy*cp, cy, sy*sp], [-sp, 0, cp]])
    # Columns are optical axes expressed in the SDF camera frame.
    sdf_from_optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    world_from_optical = np.eye(4)
    world_from_optical[:3, :3] = rotation @ sdf_from_optical
    world_from_optical[:3, 3] = position
    return {'pose_sdf_xyz_rpy': [*position.tolist(), 0., pitch, yaw],
            'world_from_optical': world_from_optical.tolist()}


def make_world(source, destination, settings):
    tree = ET.parse(source)
    world = tree.getroot().find('world')
    if world is None or world.find("model[@name='side_camera']") is not None:
        raise ValueError('Expected the unmodified official world')
    factor = world.find('physics/real_time_factor')
    if factor is not None:
        factor.text = str(settings['real_time_factor'])
    camera = settings['camera']
    model = ET.SubElement(world, 'model', name='side_camera')
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = ' '.join(map(str, camera_geometry(camera)['pose_sdf_xyz_rpy']))
    link = ET.SubElement(model, 'link', name='link')
    sensor = ET.SubElement(link, 'sensor', name='side_camera', type='camera')
    for name, value in [('always_on', 'true'), ('update_rate', settings['fps']),
                        ('topic', '/side_camera/image'), ('gz_frame_id', 'side_camera/optical')]:
        ET.SubElement(sensor, name).text = str(value)
    optics = ET.SubElement(sensor, 'camera', name='side_camera')
    ET.SubElement(optics, 'horizontal_fov').text = str(camera['horizontal_fov'])
    ET.SubElement(optics, 'camera_info_topic').text = '/side_camera/camera_info'
    ET.SubElement(optics, 'optical_frame_id').text = 'side_camera/optical'
    image = ET.SubElement(optics, 'image')
    for name, value in [('width', camera['width']), ('height', camera['height']), ('format', 'R8G8B8')]:
        ET.SubElement(image, name).text = str(value)
    clip = ET.SubElement(optics, 'clip')
    ET.SubElement(clip, 'near').text = str(camera['near'])
    ET.SubElement(clip, 'far').text = str(camera['far'])
    ET.indent(tree)
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def make_robot_description(source, camera_source, destination, camera_destination, settings, inside_run):
    """Copy official descriptions and change only camera raster size, not robot geometry."""
    ET.register_namespace('xacro', 'http://wiki.ros.org/xacro')
    camera = ET.parse(camera_source)
    for name, value in [('width', settings['record_width']), ('height', settings['record_height'])]:
        nodes = camera.getroot().findall('.//image/'+name)
        if len(nodes) != 1:
            raise ValueError('Unexpected official camera macro')
        nodes[0].text = str(value)
    camera.write(camera_destination, encoding='utf-8', xml_declaration=True)
    source_text = Path(source).read_text()
    old = '$(find aic_assets)/models/Basler Camera/basler_camera_macro.xacro'
    if source_text.count(old) != 1:
        raise ValueError('Unexpected official robot camera include')
    Path(destination).write_text(source_text.replace(old, inside_run+'/recording_camera.xacro'))


def trial_name(attempt):
    return f'episode_{attempt:06d}'


def make_engine(source, attempts, settings):
    engine = copy.deepcopy(source)
    original = source['trials']['trial_1']
    engine['trials'] = {}
    scenes = {}
    for attempt in attempts:
        trial = copy.deepcopy(original)
        # Episode zero reproduces the already validated official reference scene.
        rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], attempt]))
        pose = trial['scene']['task_board']['pose']
        if attempt:
            for coordinate, key in [('x', 'board_x_m'), ('y', 'board_y_m'),
                                    ('z', 'board_z_m'), ('yaw', 'board_yaw_rad')]:
                radius = settings['variation'][key]
                pose[coordinate] += float(rng.uniform(-radius, radius))
        for task in trial['tasks'].values():
            if (task['target_module_name'], task['plug_name'], task['port_name']) != (
                    TASK['target_module'], TASK['plug'], TASK['port']):
                raise ValueError('Collection must remain scoped to the first SFP port')
            task['time_limit'] = 90
        name = trial_name(attempt)
        trial['tasks'] = {name: next(iter(trial['tasks'].values()))}
        engine['trials'][name] = trial
        scenes[name] = trial['scene']
    return engine, scenes


def latest_causal(values, stamp, tolerance):
    """Return (timestamp, value), never a future observation; allow boundary roundoff."""
    candidates = [pair for pair in values if pair[0] <= stamp + 1e-8]
    if not candidates:
        return None
    selected = max(candidates, key=lambda pair: pair[0])
    return selected if stamp - selected[0] <= tolerance + 1e-8 else None


def causal_commands(frame_times, command_times, commands):
    times = np.asarray(command_times, dtype=float)
    commands = np.asarray(commands, dtype=float)
    if times.ndim != 1 or len(times) == 0 or commands.shape != (len(times), 7):
        raise ValueError('Missing or malformed command stream')
    if not np.isfinite(times).all() or not np.isfinite(commands).all() or np.any(np.diff(times) < 0):
        raise ValueError('Nonfinite or unordered command stream')
    indices = np.searchsorted(times, np.asarray(frame_times) + 1e-8, side='right') - 1
    valid = indices >= 0
    return indices, valid


def successful_score(score):
    return (score.get('tier_1', {}).get('score') == 1
            and score.get('tier_3', {}).get('score', 0) > 0
            and score.get('tier_3', {}).get('message') == 'Cable insertion successful.')


def validate_episode(records, commands, settings):
    if len(records) < settings['minimum_frames']:
        raise ValueError(f'Only {len(records)} frames; need {settings["minimum_frames"]}')
    times = np.array([r['simulation_time'] for r in records])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Frame timestamps must strictly increase')
    gaps = np.diff(times)
    if gaps.max() > settings['maximum_frame_gap_s']:
        raise ValueError(f'Recording gap {gaps.max():.4f}s exceeds the frame-gap limit')
    for r in records:
        for name in CAMERAS:
            age = r['simulation_time'] - r['camera_times'][name]
            if age < -1e-8 or age > settings['camera_tolerance_s'] + 1e-8:
                raise ValueError(f'Noncausal or stale {name} camera')
        for key in ('joint_time', 'wrench_time'):
            age = r['simulation_time'] - r[key]
            if age < -1e-8 or age > settings['state_tolerance_s'] + 1e-8:
                raise ValueError('Noncausal or stale state/force')
        for key, width in [('state', 6), ('arm_joint_positions', 6), ('wrench', 6), ('tcp_pose', 7)]:
            value = np.asarray(r[key])
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid {key}')
        if abs(np.linalg.norm(r['tcp_pose'][3:]) - 1) > 1e-4:
            raise ValueError('Measured TCP quaternion is not normalized')
    command_times = [c['simulation_time'] for c in commands]
    poses = np.array([c['pose'] for c in commands])
    if np.any(np.abs(np.linalg.norm(poses[:, 3:], axis=-1) - 1) > 1e-4):
        raise ValueError('Command quaternion is not normalized')
    indices, valid = causal_commands(times, command_times, poses)
    if not valid.all():
        raise ValueError('Frame has no causal command')
    return {'frames': len(records), 'duration_seconds': float(times[-1] - times[0]),
            'maximum_frame_gap_s': float(gaps.max()),
            'maximum_camera_age_s': {name: max(r['simulation_time']-r['camera_times'][name] for r in records)
                                     for name in CAMERAS},
            'command_count': len(commands), 'command_indices': indices}


def export_episode(database, destination, episode, offset, settings, *, command_provenance=False, allow_gaps=False):
    """Stream the verified SQLite recording to one self-contained Parquet episode."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/dataset_python'))
    import pyarrow as pa
    import pyarrow.parquet as pq
    connection = sqlite3.connect(f'file:{Path(database).resolve()}?mode=ro', uri=True)
    command_rows = connection.execute('SELECT metadata FROM commands ORDER BY stamp, id').fetchall()
    commands = [json.loads(row[0]) for row in command_rows]
    rows = connection.execute('SELECT metadata FROM frames ORDER BY stamp').fetchall()
    records = [json.loads(row[0]) for row in rows]
    if not commands:
        raise ValueError('No recorded commands')
    # Discard only the prefix before the first accepted command, never fill from the future.
    records = [r for r in records if r['simulation_time'] + 1e-8 >= commands[0]['simulation_time']]
    validation_settings=dict(settings,maximum_frame_gap_s=float('inf')) if allow_gaps else settings
    report = validate_episode(records, commands, validation_settings)
    indices = report.pop('command_indices')
    actions = np.array([commands[i]['pose'] for i in indices], dtype=np.float32)
    tcp_poses = np.array([r['tcp_pose'] for r in records], dtype=np.float32)
    tcp_poses[:, 3:] /= np.linalg.norm(tcp_poses[:, 3:], axis=-1, keepdims=True)
    for i in range(1, len(actions)):
        if np.dot(actions[i-1, 3:], actions[i, 3:]) < 0:
            actions[i, 3:] *= -1
        if np.dot(tcp_poses[i-1, 3:], tcp_poses[i, 3:]) < 0:
            tcp_poses[i, 3:] *= -1
    features = {f'observation.images.{name}_camera': pa.struct([('bytes', pa.binary()), ('path', pa.string())])
                for name in CAMERAS}
    widths = {'observation.state': 6, 'observation.arm_joint_positions': 6,
              'observation.environment_state': 6, 'observation.tcp_pose': 7, 'action': 7}
    features.update({name: pa.list_(pa.float32(), width) for name, width in widths.items()})
    features.update({name: pa.int64() for name in ('episode_index', 'frame_index', 'index', 'task_index')})
    features.update(timestamp=pa.float32(), simulation_time=pa.float64(), action_time=pa.float64(),
                    camera_times=pa.list_(pa.float64(), 4), joint_time=pa.float64(), wrench_time=pa.float64())
    if command_provenance:
        features.update(action_source=pa.string(),action_segment=pa.int64(),human_active=pa.bool_())
    schema = pa.schema(list(features.items()))
    hf = {}
    for name, dtype in features.items():
        if name.startswith('observation.images.'):
            hf[name] = {'_type': 'Image'}
        elif pa.types.is_fixed_size_list(dtype):
            hf[name] = {'feature': {'dtype': str(dtype.value_type).replace('float', 'float32').replace('double', 'float64'),
                                     '_type': 'Value'}, 'length': dtype.list_size, '_type': 'List'}
        else:
            hf[name] = {'dtype': str(dtype).replace('float', 'float32').replace('double', 'float64'), '_type': 'Value'}
    schema = schema.with_metadata({b'huggingface': json.dumps({'info': {'features': hf}}).encode()})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.parquet.partial')
    camera_hashes = {name: hashlib.sha256() for name in CAMERAS}
    start = records[0]['simulation_time']
    packed = 'image_offsets' in {row[1] for row in connection.execute('PRAGMA table_info(frames)')}
    with ExitStack() as stack:
        writer = stack.enter_context(pq.ParquetWriter(temporary, schema, compression='zstd'))
        images = stack.enter_context(Path(database).with_suffix('.images').open('rb')) if packed else None
        buffer = []
        columns = 'metadata,image_offsets' if packed else 'metadata,left_png,center_png,right_png,side_png'
        cursor = connection.execute(f'SELECT {columns} FROM frames ORDER BY stamp')
        frame = 0
        for raw in cursor:
            r = json.loads(raw[0])
            if r['simulation_time'] + 1e-8 < commands[0]['simulation_time']:
                continue
            if packed:
                offsets = json.loads(raw[1])
                if len(offsets)!=4: raise ValueError('Expected four PNG offsets per frame')
                pngs = []
                for image_offset,length in offsets:
                    if image_offset<0 or length<=0: raise ValueError('Invalid PNG offset/length')
                    images.seek(image_offset)
                    png=images.read(length)
                    if len(png)!=length: raise ValueError('Truncated packed PNG recording')
                    pngs.append(png)
            else:
                pngs = raw[1:]
            row = {f'observation.images.{name}_camera': {'bytes': pngs[i], 'path': None}
                   for i, name in enumerate(CAMERAS)}
            for i, name in enumerate(CAMERAS):
                camera_hashes[name].update(pngs[i])
            row.update({'observation.state': r['state'], 'observation.arm_joint_positions': r['arm_joint_positions'],
                'observation.environment_state': r['wrench'], 'observation.tcp_pose': tcp_poses[frame].tolist(),
                'action': actions[frame].tolist(), 'episode_index': episode, 'frame_index': frame,
                'index': offset+frame, 'task_index': 0, 'timestamp': r['simulation_time']-start,
                'simulation_time': r['simulation_time'], 'action_time': commands[indices[frame]]['simulation_time'],
                'camera_times': [r['camera_times'][name] for name in CAMERAS],
                'joint_time': r['joint_time'], 'wrench_time': r['wrench_time']})
            if command_provenance:
                command=commands[indices[frame]]
                row.update(action_source=command.get('control_source','unknown'),
                    action_segment=int(command.get('control_segment',-1)),human_active=bool(command.get('human_active',False)))
            buffer.append(row)
            frame += 1
            if len(buffer) == 32:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    connection.close()
    if frame != len(records):
        raise ValueError('Frame export count changed during conversion')
    temporary.replace(destination)
    report['camera_stream_sha256'] = {name: h.hexdigest() for name, h in camera_hashes.items()}
    numeric = {'observation.state': np.asarray([r['state'] for r in records]),
               'observation.arm_joint_positions': np.asarray([r['arm_joint_positions'] for r in records]),
               'observation.environment_state': np.asarray([r['wrench'] for r in records]),
               'observation.tcp_pose': tcp_poses, 'action': actions}
    report['quaternion_sign_continuity_enforced'] = True
    command_tcp_error = np.linalg.norm(actions[:, :3]-tcp_poses[:, :3], axis=-1)*1000
    report['command_vs_measured_tcp_mm'] = {'mean': float(command_tcp_error.mean()),
        'p90': float(np.quantile(command_tcp_error,.9)), 'maximum': float(command_tcp_error.max())}
    report['numeric_stats'] = {name: {'min': value.min(0).tolist(), 'max': value.max(0).tolist(),
        'mean': value.mean(0).tolist(), 'std': value.std(0).tolist(), 'count': [len(value)]}
        for name, value in numeric.items()}
    report['parquet_sha256'] = digest_file(destination)
    return report
