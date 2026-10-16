"""Consume the shared prepared configuration; do not start another model or controller manager."""
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    config = yaml.safe_load(Path(LaunchConfiguration('moveit_config_file').perform(context)).read_text())
    sim = LaunchConfiguration('use_sim_time').perform(context) == 'true'
    nodes = [Node(package='moveit_ros_move_group', executable='move_group', output='screen',
                  parameters=[config, {'use_sim_time': sim,
                  'capabilities': 'move_group/ExecuteTaskSolutionCapability'}])]
    if LaunchConfiguration('use_rviz').perform(context) == 'true':
        rviz = Path(get_package_share_directory('kinova_gen3_lite_support'))/'rviz/pick_place.rviz'
        nodes.append(Node(package='rviz2', executable='rviz2', output='screen',
                          arguments=['-d', str(rviz)], parameters=[config, {'use_sim_time': sim}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('moveit_config_file'),
                              DeclareLaunchArgument('use_sim_time', default_value='true'),
                              DeclareLaunchArgument('use_rviz', default_value='true'),
                              OpaqueFunction(function=setup)])
