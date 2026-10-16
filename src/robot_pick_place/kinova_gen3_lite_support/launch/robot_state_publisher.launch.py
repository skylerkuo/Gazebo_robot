"""Publish the exact prepared model used by Gazebo, MoveIt and MTC."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    model = Path(LaunchConfiguration('robot_description_file').perform(context)).read_text()
    sim = LaunchConfiguration('use_sim_time').perform(context) == 'true'
    return [Node(package='robot_state_publisher', executable='robot_state_publisher',
                 parameters=[{'robot_description': model, 'use_sim_time': sim}], output='screen')]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('robot_description_file'),
                              DeclareLaunchArgument('use_sim_time', default_value='true'),
                              OpaqueFunction(function=setup)])
