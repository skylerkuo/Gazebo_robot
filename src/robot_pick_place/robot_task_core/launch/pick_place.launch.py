"""Standalone task. All robot/MoveIt parameters are supplied as ROS YAML."""
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_parameters(path):
    data = yaml.safe_load(open(path, encoding='utf-8'))
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f'{path}: expected a single ROS node entry')
    return next(iter(data.values()))['ros__parameters']


def configure(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    defaults = os.path.join(get_package_share_directory('robot_task_core'), 'config', 'mtc_defaults.yaml')
    parameters = {**load_parameters(defaults), **load_parameters(value('params_file'))}
    parameters.update(scene_source=value('scene_source'), execute=value('execute') == 'true',
                      use_sim_time=value('use_sim_time') == 'true')
    return [Node(package='robot_task_core', executable='mtc_node', output='screen', parameters=[parameters])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('params_file', description='ROS YAML with robot, scene and complete MoveIt configuration'),
        DeclareLaunchArgument('scene_source', default_value='known', choices=['known', 'perception']),
        DeclareLaunchArgument('execute', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        OpaqueFunction(function=configure),
    ])
