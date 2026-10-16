"""Small Harmonic backend satisfying robot_bringup's existing simulation contract."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    get = lambda key: LaunchConfiguration(key).perform(context)
    sim = get('use_sim_time') == 'true'
    share = Path(get_package_share_directory('kinova_gen3_lite_support'))
    world = share/'worlds'/get('world_file')
    args = f'-r -v 3 {world}'
    if get('headless') == 'true':
        args = '-s --headless-rendering ' + args
    gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
        Path(get_package_share_directory('ros_gz_sim'))/'launch/gz_sim.launch.py')),
        launch_arguments={'gz_args': args}.items())
    rsp = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share/'launch/robot_state_publisher.launch.py')),
        launch_arguments={'robot_description_file': get('robot_description_file'), 'use_sim_time': get('use_sim_time')}.items())
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-world', 'default', '-name', 'kinova_gen3_lite', '-allow_renaming', 'false',
                            '-file', get('robot_description_file')], parameters=[{'use_sim_time': sim}])
    controllers = [Node(package='controller_manager', executable='spawner', output='screen',
        arguments=[name, '--controller-manager', '/controller_manager', '--controller-manager-timeout', '90',
                   '--param-file', get('controllers_file')], parameters=[{'use_sim_time': sim}])
        for name in ['joint_state_broadcaster', 'joint_trajectory_controller', 'gen3_lite_2f_gripper_controller']]

    def next_step(action):
        def callback(event, context):
            return [action] if event.returncode == 0 else [EmitEvent(event=Shutdown(reason='Kinova model/controller startup failed'))]
        return callback

    handlers = [RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=next_step(controllers[0])))]
    for current, following in zip(controllers, controllers[1:]):
        handlers.append(RegisterEventHandler(OnProcessExit(target_action=current, on_exit=next_step(following))))
    def last_controller(event, context):
        return [] if event.returncode == 0 else [EmitEvent(event=Shutdown(reason='Kinova gripper controller failed'))]
    handlers.append(RegisterEventHandler(OnProcessExit(target_action=controllers[-1], on_exit=last_controller)))
    bridges = ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    remaps = []
    if get('use_camera') == 'true':
        bridges += ['/camera_head/image@sensor_msgs/msg/Image[gz.msgs.Image',
                    '/camera_head/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked']
        remaps = [('/camera_head/image', '/camera_head/color/image_raw'),
                  ('/camera_head/points', '/camera_head/depth/color/points')]
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', arguments=bridges,
                  remappings=remaps, parameters=[{'use_sim_time': sim}], output='screen')
    return [AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', str(Path(get_package_share_directory('kortex_description')).parent)),
            *handlers, rsp, gazebo, bridge, spawn]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_description_file'), DeclareLaunchArgument('controllers_file'),
        DeclareLaunchArgument('world_file', default_value='pick_place.sdf'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_camera', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        OpaqueFunction(function=setup),
    ])
