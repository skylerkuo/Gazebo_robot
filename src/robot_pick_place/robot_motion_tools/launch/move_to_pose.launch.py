import atexit
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_bringup.configuration import prepare_configuration
from robot_bringup.profile import load_profile


def setup(context):
    def get(name):
        return LaunchConfiguration(name).perform(context)

    profile = load_profile(get('robot'))

    # 沿用既有工具，在暫存目錄準備相同模型與設定。
    runtime = tempfile.TemporaryDirectory(prefix='move_to_pose_')
    atexit.register(runtime.cleanup)

    moveit, _, _, _, _ = prepare_configuration(
        profile, runtime.name, 'known', False)

    planning = profile['planning']

    return [
        Node(
            package='robot_motion_tools',
            executable='move_to_pose_node',
            output='screen',
            parameters=[
                moveit,
                {
                    'use_sim_time': True,
                    'execute': get('execute') == 'true',
                    'arm_group': planning['arm_group'],
                    'reference_frame': planning['base_frame'],
                    'end_effector_link': planning['ik_frame'],
                    'planning_pipeline': planning['pipeline'],
                    'planner_id': planning['planner_id'],
                    'target_x': float(get('x')),
                    'target_y': float(get('y')),
                    'target_z': float(get('z')),
                    'roll_deg': float(get('roll')),
                    'pitch_deg': float(get('pitch')),
                    'yaw_deg': float(get('yaw')),
                },
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot', default_value='kinova_gen3_lite'),
        DeclareLaunchArgument('x', default_value='0.30'),
        DeclareLaunchArgument('y', default_value='0.10'),
        DeclareLaunchArgument('z', default_value='0.40'),
        DeclareLaunchArgument('roll', default_value='0.0'),
        DeclareLaunchArgument('pitch', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument(
            'execute',
            default_value='false',
            choices=['true', 'false']),
        OpaqueFunction(function=setup),
    ])