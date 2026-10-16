#!/usr/bin/env python3
"""
Launch RViz visualization for the mycobot robot.

This launch file sets up the complete visualization environment for the mycobot robot,
including robot state publisher, joint state publisher, and RViz2. It handles loading
and processing of URDF/XACRO files and controller configurations.

:author: Addison Sears-Collins
:date: November 15, 2024
"""
import os
from pathlib import Path
import atexit
import shutil
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def process_ros2_controllers_config(context):
    """Render the package template before evaluating robot_description's Xacro."""
    existing = LaunchConfiguration('controllers_file').perform(context)
    if existing:
        if not Path(existing).is_file():
            raise FileNotFoundError(f'Controller configuration does not exist: {existing}')
        return []
    prefix = LaunchConfiguration('prefix').perform(context)
    flange_link = LaunchConfiguration('flange_link').perform(context)
    robot_name = LaunchConfiguration('robot_name').perform(context)
    template_path = os.path.join(
        get_package_share_directory('mycobot_moveit_config'), 'config',
        robot_name, 'ros2_controllers_template.yaml')
    with open(template_path, encoding='utf-8') as file:
        content = file.read().replace('${prefix}', prefix).replace('${flange_link}', flange_link)

    # Each launch owns its runtime file; source and install trees remain read-only.
    runtime_dir = tempfile.mkdtemp(prefix='mycobot_controllers_')
    output_path = os.path.join(runtime_dir, 'ros2_controllers.yaml')
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)

    # Gazebo may still read this file while launch is shutting down.
    atexit.register(shutil.rmtree, runtime_dir, ignore_errors=True)

    return []

    return [
        SetLaunchConfiguration('controllers_file', output_path),
        LogInfo(msg=['[Robot Description] Controller configuration: ', output_path]),
    ]


# Define the arguments for the XACRO file
ARGUMENTS = [
    DeclareLaunchArgument('controllers_file', default_value='',
                          description='Optional prepared runtime controller YAML'),
    DeclareLaunchArgument('robot_description_file', default_value='',
                          description='Optional prepared URDF shared with MoveIt and MTC'),
    DeclareLaunchArgument('robot_name', default_value='mycobot_280',
                          description='Name of the robot'),
    DeclareLaunchArgument('prefix', default_value='',
                          description='Prefix for robot joints and links'),
    DeclareLaunchArgument('add_world', default_value='true',
                          choices=['true', 'false'],
                          description='Whether to add world link'),
    DeclareLaunchArgument('base_link', default_value='base_link',
                          description='Name of the base link'),
    DeclareLaunchArgument('base_type', default_value='g_shape',
                          description='Type of the base'),
    DeclareLaunchArgument('flange_link', default_value='link6_flange',
                          description='Name of the flange link'),
    DeclareLaunchArgument('gripper_type', default_value='adaptive_gripper',
                          description='Type of the gripper'),
    DeclareLaunchArgument('use_camera', default_value='false',
                          choices=['true', 'false'],
                          description='Whether to use the RGBD Gazebo plugin for point cloud'),
    DeclareLaunchArgument('use_gazebo', default_value='false',
                          choices=['true', 'false'],
                          description='Whether to use Gazebo simulation'),
    DeclareLaunchArgument('use_gripper', default_value='true',
                          choices=['true', 'false'],
                          description='Whether to attach a gripper')
]


def generate_launch_description():
    """Generate the launch description for the mycobot robot visualization.

    This function sets up all necessary nodes and parameters for visualizing
    the mycobot robot in RViz, including:
    - Robot state publisher for broadcasting transforms
    - Joint state publisher for simulating joint movements
    - RViz for visualization

    Returns:
        LaunchDescription: Complete launch description for the visualization setup
    """
    # Define filenames
    urdf_package = 'mycobot_description'
    urdf_filename = 'mycobot_280.urdf.xacro'
    rviz_config_filename = 'mycobot_280_description.rviz'

    # Set paths to important files
    pkg_share_description = FindPackageShare(urdf_package)
    default_urdf_model_path = PathJoinSubstitution(
        [pkg_share_description, 'urdf', 'robots', urdf_filename])
    default_rviz_config_path = PathJoinSubstitution(
        [pkg_share_description, 'rviz', rviz_config_filename])

    # Launch configuration variables
    jsp_gui = LaunchConfiguration('jsp_gui')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    urdf_model = LaunchConfiguration('urdf_model')
    use_jsp = LaunchConfiguration('use_jsp')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare the launch arguments
    declare_jsp_gui_cmd = DeclareLaunchArgument(
        name='jsp_gui',
        default_value='true',
        choices=['true', 'false'],
        description='Flag to enable joint_state_publisher_gui')

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        name='rviz_config_file',
        default_value=default_rviz_config_path,
        description='Full path to the RVIZ config file to use')

    declare_urdf_model_path_cmd = DeclareLaunchArgument(
        name='urdf_model',
        default_value=default_urdf_model_path,
        description='Absolute path to robot urdf file')

    declare_use_jsp_cmd = DeclareLaunchArgument(
        name='use_jsp',
        default_value='false',
        choices=['true', 'false'],
        description='Enable the joint state publisher')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
        description='Whether to start RVIZ')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    xacro_description = ParameterValue(Command([
        'xacro', ' ', urdf_model, ' ',
        'robot_name:=', LaunchConfiguration('robot_name'), ' ',
        'prefix:=', LaunchConfiguration('prefix'), ' ',
        'add_world:=', LaunchConfiguration('add_world'), ' ',
        'base_link:=', LaunchConfiguration('base_link'), ' ',
        'base_type:=', LaunchConfiguration('base_type'), ' ',
        'flange_link:=', LaunchConfiguration('flange_link'), ' ',
        'gripper_type:=', LaunchConfiguration('gripper_type'), ' ',
        'use_camera:=', LaunchConfiguration('use_camera'), ' ',
        'use_gazebo:=', LaunchConfiguration('use_gazebo'), ' ',
        'use_gripper:=', LaunchConfiguration('use_gripper'), ' ',
        "controllers_file:='", LaunchConfiguration('controllers_file'), "'"
    ]), value_type=str)

    def resolve_description(context):
        path = LaunchConfiguration('robot_description_file').perform(context)
        content = Path(path).read_text(encoding='utf-8') if path else xacro_description.evaluate(context)
        return [SetLaunchConfiguration('resolved_robot_description', content)]

    robot_description_content = ParameterValue(LaunchConfiguration('resolved_robot_description'), value_type=str)

    # Subscribe to the joint states of the robot, and publish the 3D pose of each link.
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description_content}])

    # Publish the joint state values for the non-fixed joints in the URDF file.
    start_joint_state_publisher_cmd = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_jsp))

    # Depending on gui parameter, either launch joint_state_publisher or joint_state_publisher_gui
    start_joint_state_publisher_gui_cmd = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(jsp_gui))

    # Launch RViz
    start_rviz_cmd = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}])

    # Create the launch description and populate
    ld = LaunchDescription(ARGUMENTS)

    # Process the controller configuration before starting nodes
    ld.add_action(OpaqueFunction(function=process_ros2_controllers_config))

    # Declare the launch options
    ld.add_action(declare_jsp_gui_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_urdf_model_path_cmd)
    ld.add_action(declare_use_jsp_cmd) 
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_use_sim_time_cmd)

    ld.add_action(OpaqueFunction(function=resolve_description))

    # Add any actions
    ld.add_action(start_joint_state_publisher_cmd)
    ld.add_action(start_joint_state_publisher_gui_cmd)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_rviz_cmd)

    return ld
