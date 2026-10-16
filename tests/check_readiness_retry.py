"""Real ROS service regression: a stalled ListControllers request must not pin readiness.

Runs a mock manager in its own ROS domain. No robot or Gazebo is controlled.
"""
import os
import subprocess
import sys
import threading

os.environ['ROS_DOMAIN_ID'] = str(220 + os.getpid() % 10)
os.environ['ROS_LOG_DIR'] = '/tmp/readiness-retry-test-logs'

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import TransformStamped
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster


def main():
    rclpy.init()
    node = rclpy.create_node('readiness_mock_manager')
    release = threading.Event()
    state = {'requests': 0, 'hold_first': True, 'controller_state': 'active'}

    def controllers(request, response):
        state['requests'] += 1
        if state['hold_first'] and state['requests'] == 1:
            release.wait(timeout=15.0)
        response.controller = [ControllerState(name='arm_controller', state=state['controller_state'])]
        return response

    node.create_service(ListControllers, '/controller_manager/list_controllers', controllers,
                        callback_group=ReentrantCallbackGroup())
    node.create_service(ApplyPlanningScene, '/apply_planning_scene', lambda req, res: res)
    node.create_service(GetPlanningScene, '/get_planning_scene', lambda req, res: res)
    publisher = node.create_publisher(JointState, '/joint_states', 10)

    def publish():
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name, msg.position = ['joint_1'], [0.0]
        publisher.publish(msg)

    node.create_timer(0.05, publish)
    tf = StaticTransformBroadcaster(node)
    transform = TransformStamped()
    transform.header.frame_id = 'base_link'
    transform.child_frame_id = 'tool_link'
    transform.transform.rotation.w = 1.0
    tf.sendTransform(transform)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()

    def check(timeout):
        return subprocess.run([
            sys.executable, '-c', 'from robot_bringup.readiness import main; raise SystemExit(main())',
            '--ros-args', '-p', 'use_sim_time:=false', '-p', f'timeout:={timeout}',
            '-p', 'controller_names:=[arm_controller]', '-p', 'joint_names:=[joint_1]',
            '-p', 'base_frame:=base_link', '-p', 'ik_frame:=tool_link'],
            text=True, capture_output=True, timeout=12)

    try:
        recovered = check(8.0)
        print(recovered.stdout + recovered.stderr)
        assert recovered.returncode == 0, 'Readiness did not recover while first response was held'
        assert state['requests'] >= 2 and not release.is_set()
        assert 'did not reply within 3s; retrying' in recovered.stderr + recovered.stdout
        print('Passed: stalled controller query retried and readiness recovered before first reply')
        release.set()
        state['hold_first'] = False
        state['controller_state'] = 'inactive'
        rejected = check(2.0)
        assert rejected.returncode == 1, rejected.stdout + rejected.stderr
        assert 'controllers active: arm_controller' in rejected.stdout + rejected.stderr
        print('Passed: inactive controller still blocks readiness and times out')
    finally:
        release.set()
        executor.shutdown(timeout_sec=5.0)
        thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
