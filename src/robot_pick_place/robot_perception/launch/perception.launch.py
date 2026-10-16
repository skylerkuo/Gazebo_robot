"""Standalone generic perception; robot_bringup supplies sensor/frame overrides."""
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def configure(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    with open(value('params_file'), encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError('params_file must contain one ROS node entry')
    parameters = next(iter(data.values()))['ros__parameters']
    parameters.update(use_sim_time=value('use_sim_time') == 'true', service_name=value('service_name'))
    return [Node(package='robot_perception', executable='get_planning_scene_server',
                 output='screen', parameters=[parameters])]


def generate_launch_description():
    defaults = os.path.join(get_package_share_directory('robot_perception'), 'config', 'perception_defaults.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=defaults),
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('service_name', default_value='/perception/get_planning_scene'),
        OpaqueFunction(function=configure),
    ])
