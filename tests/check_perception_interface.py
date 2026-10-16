"""Exercise the installed generic service, including the real no-sensor failure path."""
import os
import signal
import subprocess
import tempfile
import time
import rclpy
from robot_task_interfaces.srv import GetPlanningScene
from ament_index_python.packages import get_package_prefix


def main():
    rclpy.init()
    node = rclpy.create_node('perception_contract_test')
    executable = os.path.join(get_package_prefix('robot_perception'), 'lib',
                              'robot_perception', 'get_planning_scene_server')
    with tempfile.TemporaryFile(mode='w+') as output:
        process = subprocess.Popen([executable, '--ros-args', '-p', 'use_sim_time:=false',
                                    '-p', 'service_name:=/test_scene'], stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            client = node.create_client(GetPlanningScene, '/test_scene')
            assert client.wait_for_service(timeout_sec=15.0), 'Generic service did not become ready'
            request = GetPlanningScene.Request(target_shape='cylinder', target_dimensions=[0.35, 0.0125])
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
            assert future.done(), 'Service response timed out'
            assert future.result().success is False, 'No sensor data must return success=false'
            assert not future.result().scene_world.collision_objects
            print('Passed: robot_task_interfaces client/server round trip; missing sensors returns failure')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
            output.seek(0)
            log = output.read()
            assert 'Point cloud or RGB image data not available' in log, log[-3000:]
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
