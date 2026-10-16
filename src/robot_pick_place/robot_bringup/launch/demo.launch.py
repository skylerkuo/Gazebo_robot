"""One robot profile, one simulation, one move_group, and one reusable MTC task."""
import atexit
import shutil
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, GroupAction, IncludeLaunchDescription, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_bringup.configuration import prepare_configuration
from robot_bringup.profile import load_profile, load_ros_parameters, package_file


def boolean(value, name):
    if value.lower() not in ('true', 'false'):
        raise ValueError(f'{name} must be true or false, got {value!r}')
    return value.lower() == 'true'


def setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context)
    profile = load_profile(get('robot'), get('profile'))
    source = get('scene_source')
    if source not in ('known', 'perception'):
        raise ValueError(f'scene_source must be known or perception, got {source!r}')
    execute = boolean(get('execute'), 'execute')
    sim_time = boolean(get('use_sim_time'), 'use_sim_time')
    camera = source == 'perception' if not get('use_camera') else boolean(get('use_camera'), 'use_camera')
    if camera != (source == 'perception'):
        raise ValueError('use_camera must be false for known and true for perception')
    timeout = float(get('service_timeout'))
    readiness_timeout = float(get('readiness_timeout'))
    if not (0.0 < timeout < float('inf') and 0.0 < readiness_timeout < float('inf')):
        raise ValueError('service_timeout and readiness_timeout must be positive finite seconds')
    runtime = tempfile.mkdtemp(prefix='robot_task_')

    try:
        moveit, task, perception, urdf_file, controllers_file = prepare_configuration(profile, runtime, source, camera)
        task.update(use_sim_time=sim_time, scene_source=source, execute=execute, service_timeout=timeout)
        if get('task_params_file'):
            task.update(load_ros_parameters(get('task_params_file')))
            # Safety and launch mode remain explicit launch options.
            task.update(use_sim_time=sim_time, scene_source=source, execute=execute, service_timeout=timeout)
        moveit_file = Path(runtime) / 'moveit.yaml'
        moveit_file.write_text(yaml.safe_dump(moveit, sort_keys=False))
        defaults = package_file('robot_task_core', 'config/mtc_defaults.yaml', 'task defaults')
        demo = Node(package='robot_task_core', executable='mtc_node', output='screen',
                    parameters=[{**moveit, **load_ros_parameters(defaults), **task}])
        controller_map = moveit['moveit_simple_controller_manager']
        joint_names = sorted({joint for name in task['controller_names'] for joint in controller_map[name]['joints']})
        ready = Node(package='robot_bringup', executable='wait_ready', output='screen', parameters=[{
            'use_sim_time': sim_time, 'timeout': readiness_timeout,
            'controller_manager': profile['execution']['controller_manager'],
            'controller_names': task['controller_names'] + [profile['execution']['state_controller']],
            'joint_names': joint_names, 'base_frame': task['world_frame'], 'ik_frame': task['gripper_frame'],
            'scene_source': source, 'scene_service_name': task['scene_service_name'],
            'point_cloud_topic': perception['point_cloud_topic'], 'rgb_image_topic': perception['rgb_image_topic'],
        }])

        def after_ready(event, context):
            if event.returncode != 0:
                return [LogInfo(msg='[Bringup] Readiness failed; task was not started'),
                        EmitEvent(event=Shutdown(reason='Robot readiness failed'))]
            return [demo]

        def after_task(event, context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(reason=f'MTC failed with code {event.returncode}'))]
            return []

        simulation = profile['simulation']
        arguments = {**simulation['arguments'], 'robot_name': profile['moveit']['robot_name'],
                     'world_file': get('world_file') or simulation['world_file'],
                     'use_camera': str(camera).lower(), 'headless': get('headless'),
                     'use_rviz': 'false', 'jsp_gui': 'false', 'use_robot_state_pub': 'true',
                     'use_sim_time': str(sim_time).lower(), 'load_controllers': 'true',
                     'robot_description_file': str(urdf_file), 'controllers_file': str(controllers_file)}
        gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(package_file(
            simulation['package'], 'launch/' + simulation['launch_file'], 'simulation.launch_file'))),
            launch_arguments=arguments.items())
        config = profile['moveit']
        move_group = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(package_file(
            config['package'], 'launch/' + config['launch_file'], 'moveit.launch_file'))),
            launch_arguments={'robot_name': config['robot_name'], 'use_sim_time': str(sim_time).lower(),
                              'use_rviz': get('use_rviz'), config['config_argument']: str(moveit_file)}.items())
        actions = [

            RegisterEventHandler(OnProcessExit(target_action=ready, on_exit=after_ready)),
            RegisterEventHandler(OnProcessExit(target_action=demo, on_exit=after_task)),
            LogInfo(msg=f'[Bringup] profile={profile["robot_id"]}, scene_source={source}, execute={execute}, runtime={runtime}'),
            # Vendor launches reuse names such as rviz_config_file and use_rviz.
            # Keep their defaults local so Gazebo/RSP cannot overwrite MoveIt's RViz config.
            GroupAction(scoped=True, actions=[gazebo]),
            GroupAction(scoped=True, actions=[move_group]),
        ]
        if source == 'perception':
            actions.append(Node(package='robot_perception', executable='get_planning_scene_server',
                                output='screen',
                                parameters=[perception, {'use_sim_time': sim_time}]))
        actions.append(ready)
        # Keep resources until launch and its managed processes have actually exited.
        atexit.register(shutil.rmtree, runtime, ignore_errors=True)
        return actions
    except Exception:
        shutil.rmtree(runtime, ignore_errors=True)
        raise


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot', default_value='mycobot_280', description='Installed robot profile ID'),
        DeclareLaunchArgument('profile', default_value='', description='Optional external profile YAML (robot_id must match robot)'),
        DeclareLaunchArgument('scene_source', default_value='perception', choices=['known', 'perception']),
        DeclareLaunchArgument('execute', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('use_rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false'], description='Gazebo server only with headless sensor rendering'),
        DeclareLaunchArgument('world_file', default_value='', description='World resource name; empty uses profile'),
        DeclareLaunchArgument('use_camera', default_value='', description='Auto from scene_source; known=false, perception=true'),
        DeclareLaunchArgument('service_timeout', default_value='30.0'),
        DeclareLaunchArgument('readiness_timeout', default_value='120.0'),
        DeclareLaunchArgument('task_params_file', default_value='', description='Optional ROS YAML scene/task overrides'),
        OpaqueFunction(function=setup),
    ])
