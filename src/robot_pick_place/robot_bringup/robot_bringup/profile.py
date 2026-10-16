"""Version 1 robot profile contract. Resource locations always use the ament index."""
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


class ProfileError(ValueError):
    pass


def require(mapping, key, kind, location):
    if not isinstance(mapping, dict) or key not in mapping or not isinstance(mapping[key], kind):
        raise ProfileError(f'{location}.{key}: missing or expected {kind.__name__}')
    value = mapping[key]
    if isinstance(value, str) and not value.strip():
        raise ProfileError(f'{location}.{key}: must not be empty')
    return value


def package_file(package, relative, field):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '~' in relative or '..' in Path(relative).parts:
        raise ProfileError(f'{field}: expected a nonempty package-relative resource path')
    try:
        path = Path(get_package_share_directory(package)) / relative
    except PackageNotFoundError as error:
        raise ProfileError(f'{field}: package {package!r} is not installed/sourced') from error
    if not path.is_file():
        raise ProfileError(f'{field}: file does not exist: {package}/{relative}')
    return path


def resource(reference, field):
    return package_file(require(reference, 'package', str, field),
                        require(reference, 'file', str, field), field)


def load_ros_parameters(path):
    """Read a single-node ROS YAML independent of its legacy node name."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or len(data) != 1:
        raise ProfileError(f'{path}: expected exactly one node entry')
    entry = next(iter(data.values()))
    return require(entry, 'ros__parameters', dict, str(path))


def load_profile(robot, profile_file=''):
    if not re.fullmatch(r'[a-zA-Z0-9_]+', robot):
        raise ProfileError(f'robot={robot!r}: expected a profile ID')
    path = Path(profile_file) if profile_file else package_file(
        'robot_bringup', f'config/robots/{robot}.yaml', f'robot={robot}')
    if not path.is_file():
        raise ProfileError(f'profile: file does not exist: {path}')
    profile = yaml.safe_load(path.read_text())
    if require(profile, 'schema_version', int, 'profile') != 1:
        raise ProfileError('profile.schema_version: only version 1 is supported')
    if require(profile, 'robot_id', str, 'profile') != robot:
        raise ProfileError(f'profile.robot_id must match robot={robot!r}')
    for section in ('simulation', 'robot_description', 'moveit'):
        value = require(profile, section, dict, 'profile')
        package_file(require(value, 'package', str, section),
                     'launch/' + require(value, 'launch_file', str, section), section + '.launch_file')
    description = profile['robot_description']
    package_file(description['package'], require(description, 'xacro_file', str, 'robot_description'), 'robot_description.xacro_file')
    mappings = require(description, 'mappings', dict, 'robot_description')
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in mappings.items()):
        raise ProfileError('robot_description.mappings: expected string keys/values (quote booleans)')
    simulation = profile['simulation']
    package_file(simulation['package'], 'worlds/' + require(simulation, 'world_file', str, 'simulation'), 'simulation.world_file')
    require(simulation, 'arguments', dict, 'simulation')
    moveit = profile['moveit']
    for field in ('robot_name', 'config_argument'):
        require(moveit, field, str, 'moveit')
    files = require(moveit, 'files', dict, 'moveit')
    for field in ('semantic', 'kinematics', 'joint_limits', 'controllers', 'ros2_controllers', 'controllers_template', 'cartesian_limits'):
        package_file(moveit['package'], require(files, field, str, 'moveit.files'), 'moveit.files.' + field)
    pipelines = require(moveit, 'pipelines', dict, 'moveit')
    if not pipelines:
        raise ProfileError('moveit.pipelines must not be empty')
    for name, relative in pipelines.items():
        package_file(moveit['package'], relative, 'moveit.pipelines.' + name)
    planning = require(profile, 'planning', dict, 'profile')
    for field in ('arm_group', 'gripper_group', 'end_effector', 'ik_frame', 'base_frame', 'home_pose', 'open_pose', 'close_pose', 'pipeline', 'planner_id'):
        require(planning, field, str, 'planning')
    transform = require(planning, 'grasp_frame_transform', list, 'planning')
    if len(transform) not in (6, 7):
        raise ProfileError('planning.grasp_frame_transform: requires xyz+rpy or xyz+xyzw')
    if planning['pipeline'] not in pipelines:
        raise ProfileError('planning.pipeline is absent from moveit.pipelines')
    execution = require(profile, 'execution', dict, 'profile')
    for field in ('controller_manager', 'state_controller', 'arm_controller', 'gripper_controller', 'gripper_strategy'):
        require(execution, field, str, 'execution')
    if execution['gripper_strategy'] != 'named_pose':
        raise ProfileError('execution.gripper_strategy: only named_pose is implemented; provide an adapter for other strategies')
    perception = require(profile, 'perception', dict, 'profile')
    for field in ('point_cloud_topic', 'rgb_image_topic', 'target_frame', 'service_name'):
        require(perception, field, str, 'perception')
    resource(require(perception, 'params', dict, 'perception'), 'perception.params')
    for source in ('known', 'perception'):
        resource(require(require(profile, 'scene', dict, 'profile'), source, dict, 'scene'), 'scene.' + source)
    validate_srdf_and_controllers(profile)
    return profile


def validate_srdf_and_controllers(profile):
    moveit, planning, execution = (profile[key] for key in ('moveit', 'planning', 'execution'))
    path = lambda field: package_file(moveit['package'], moveit['files'][field], 'moveit.files.' + field)
    srdf = ET.parse(path('semantic')).getroot()
    groups = {group.attrib['name'] for group in srdf.findall('group')}
    for field in ('arm_group', 'gripper_group'):
        if planning[field] not in groups:
            raise ProfileError(f'planning.{field}={planning[field]!r}: SRDF groups are {sorted(groups)}')
    for field, group in (('home_pose', 'arm_group'), ('open_pose', 'gripper_group'), ('close_pose', 'gripper_group')):
        states = [s.attrib['name'] for s in srdf.findall('group_state') if s.attrib['group'] == planning[group]]
        if planning[field] not in states:
            raise ProfileError(f'planning.{field}={planning[field]!r}: available states for {planning[group]}: {states}')
    eef = next((e for e in srdf.findall('end_effector') if e.attrib['name'] == planning['end_effector']), None)
    if eef is None or eef.attrib['group'] != planning['gripper_group']:
        raise ProfileError('planning.end_effector: SRDF end effector does not refer to planning.gripper_group')
    moveit_controllers = yaml.safe_load(path('controllers').read_text())['moveit_simple_controller_manager']
    ros2_controllers = yaml.safe_load(path('ros2_controllers').read_text())
    manager = ros2_controllers['controller_manager']['ros__parameters']
    for field, allowed in (('arm_controller', 'FollowJointTrajectory'), ('gripper_controller', 'GripperCommand')):
        name = execution[field]
        if name not in moveit_controllers.get('controller_names', []) or name not in manager:
            raise ProfileError(f'execution.{field}={name!r}: missing from MoveIt or ros2_control controllers')
        if moveit_controllers[name].get('type') != allowed:
            raise ProfileError(f'execution.{field}: this backend requires {allowed}; use a controller/action adapter for other types')
        joints = moveit_controllers[name].get('joints', [])
        params = ros2_controllers[name]['ros__parameters']
        hardware_joints = params.get('joints', [params.get('joint')])
        if not joints or set(joints) != set(hardware_joints):
            raise ProfileError(f'execution.{field}: MoveIt joints {joints} differ from ros2_control joints {hardware_joints}')


def task_parameters(profile):
    planning, execution = profile['planning'], profile['execution']
    mapping = {'arm_group': 'arm_group_name', 'gripper_group': 'gripper_group_name',
               'end_effector': 'end_effector', 'ik_frame': 'gripper_frame', 'base_frame': 'world_frame',
               'home_pose': 'arm_home_pose', 'open_pose': 'gripper_open_pose', 'close_pose': 'gripper_close_pose',
               'grasp_frame_transform': 'grasp_frame_transform', 'pipeline': 'planning_pipeline', 'planner_id': 'planner_id'}
    values = {parameter: planning[field] for field, parameter in mapping.items()}
    values.update(controller_names=[execution['arm_controller'], execution['gripper_controller']],
                  controller_manager=execution['controller_manager'], gripper_strategy=execution['gripper_strategy'],
                  scene_service_name=profile['perception']['service_name'])
    return values
