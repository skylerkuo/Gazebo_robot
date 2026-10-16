#!/usr/bin/env python3
"""Activate the original controllers sequentially when their manager is ready."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def configure(context):
    # Process-exit handlers run after an include scope has been popped. Resolve
    # these values now so later spawners do not depend on that temporary context.
    manager = LaunchConfiguration('controller_manager').perform(context)
    timeout = LaunchConfiguration('controller_timeout').perform(context)
    sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'

    def spawner(name):
        return Node(package='controller_manager', executable='spawner', output='screen',
                    arguments=[name, '--controller-manager', manager,
                               '--controller-manager-timeout', timeout],
                    parameters=[{'use_sim_time': sim_time}])

    broadcaster = spawner('joint_state_broadcaster')
    arm = spawner('arm_controller')
    gripper = spawner('gripper_action_controller')

    def after(next_action):
        def handle(event, context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(reason='Controller activation failed; check controller manager logs'))]
            return [next_action] if next_action is not None else []
        return handle

    return [
        RegisterEventHandler(OnProcessExit(target_action=broadcaster, on_exit=after(arm))),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=after(gripper))),
        RegisterEventHandler(OnProcessExit(target_action=gripper, on_exit=after(None))),
        broadcaster,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('controller_manager', default_value='/controller_manager'),
        DeclareLaunchArgument('controller_timeout', default_value='60'),
        OpaqueFunction(function=configure),
    ])
