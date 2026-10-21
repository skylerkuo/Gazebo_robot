"""Bounded readiness checks; exit 0 lets launch start the task, exit 1 shuts it down."""
import time

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from robot_task_interfaces.srv import GetPlanningScene as PerceptionScene
from sensor_msgs.msg import JointState, PointCloud2, Image
from tf2_ros import Buffer, TransformListener
from rclpy.qos import qos_profile_sensor_data


def main(args=None):
    rclpy.init(args=args)
    node = Node('robot_task_readiness')
    defaults = {'controller_manager': '/controller_manager', 'controller_names': [''],
                'joint_names': [''], 'base_frame': '', 'ik_frame': '', 'timeout': 120.0,
                'scene_source': 'known', 'scene_service_name': '/perception/get_planning_scene',
                'point_cloud_topic': '/camera/depth/points', 'rgb_image_topic': '/camera/color/image_raw'}
    defaults['required_image_topics'] = ['']
    for name, value in defaults.items():
        node.declare_parameter(name, value)
    value = lambda name: node.get_parameter(name).value
    buffer = Buffer(node=node)
    listener = TransformListener(buffer, node, spin_thread=False)
    joints = set()
    sensors = set()
    subscriptions = [node.create_subscription(JointState, '/joint_states',
                     lambda msg: joints.update(msg.name) if msg.position else None, 10)]
    image_counts = {}
    for topic in filter(None, value('required_image_topics')):
        image_counts[topic] = 0
        def image_ready(msg, topic=topic):
            if msg.width > 0 and msg.height > 0 and msg.data:
                image_counts[topic] += 1
        subscriptions.append(node.create_subscription(Image, topic, image_ready, qos_profile_sensor_data))
    if value('scene_source') == 'perception':
        subscriptions.extend([
            node.create_subscription(PointCloud2, value('point_cloud_topic'),
                                     lambda msg: sensors.add('pointcloud') if msg.data else None, 10),
            node.create_subscription(Image, value('rgb_image_topic'),
                                     lambda msg: sensors.add('image') if msg.data else None, 10),
        ])
    controller = node.create_client(ListControllers, value('controller_manager') + '/list_controllers')
    services = [node.create_client(ApplyPlanningScene, '/apply_planning_scene'),
                node.create_client(GetPlanningScene, '/get_planning_scene')]
    if value('scene_source') == 'perception':
        services.append(node.create_client(PerceptionScene, value('scene_service_name')))
    deadline = time.monotonic() + value('timeout')
    next_request = 0.0
    next_log = 0.0
    future = None
    request_started = 0.0
    request_timeout = 3.0
    active = set()
    missing = []
    success = False
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if future is not None and future.done():
                try:
                    result = future.result()
                    active = {c.name for c in result.controller if c.state == 'active'} if result else set()
                except Exception as error:
                    active = set()
                    node.get_logger().warning(f'[Readiness] Controller query failed; retrying: {error}')
                future = None
            if future is not None and now - request_started >= request_timeout:
                # A response can be delayed/lost while the manager is starting. Do not pin
                # readiness to that single request for the entire global timeout.
                controller.remove_pending_request(future)
                future.cancel()
                future = None
                active = set()
                node.get_logger().warning(
                    f'[Readiness] {controller.srv_name} did not reply within {request_timeout:.0f}s; retrying')
            if future is None and controller.service_is_ready() and now >= next_request:
                future = controller.call_async(ListControllers.Request())
                request_started = now
                next_request = now + 1.0
            missing = []
            inactive = set(value('controller_names')) - active
            if inactive:
                missing.append('controllers active: ' + ', '.join(sorted(inactive)))
            absent = set(value('joint_names')) - joints
            if absent:
                missing.append('joint_states: ' + ', '.join(sorted(absent)))
            if not buffer.can_transform(value('base_frame'), value('ik_frame'), rclpy.time.Time()):
                missing.append(f"TF {value('base_frame')} <- {value('ik_frame')}")
            if node.get_parameter('use_sim_time').value and node.get_clock().now().nanoseconds == 0:
                missing.append('/clock')
            for service in services:
                if not service.service_is_ready():
                    missing.append('service ' + service.srv_name)
            if value('scene_source') == 'perception' and sensors != {'pointcloud', 'image'}:
                missing.append('nonempty pointcloud/RGB image')
            for topic, count in image_counts.items():
                if count < 2:
                    missing.append('two RGB frames on ' + topic)
            if not missing:
                node.get_logger().info('[Readiness] Controllers, joint states, TF, clock and scene services are ready')
                success = True
                break
            if now >= next_log:
                node.get_logger().info('[Readiness] Waiting for ' + '; '.join(missing))
                next_log = now + 5.0
        if not success:
            node.get_logger().error('[Readiness] Timeout; missing: ' + '; '.join(missing))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if success else 1
