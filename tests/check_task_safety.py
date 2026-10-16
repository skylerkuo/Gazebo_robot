"""Run the real task executable with real robot config and isolated service failure cases."""
import copy
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import yaml
import rclpy
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from moveit_configs_utils import MoveItConfigsBuilder
from robot_task_interfaces.srv import GetPlanningScene


def main():
    config = Path(get_package_share_directory('mycobot_moveit_config')) / 'config/mycobot_280'
    moveit = (MoveItConfigsBuilder('mycobot_280', package_name='mycobot_moveit_config')
              .robot_description_semantic(file_path=str(config / 'mycobot_280.srdf'))
              .robot_description_kinematics(file_path=str(config / 'kinematics.yaml'))
              .joint_limits(file_path=str(config / 'joint_limits.yaml'))
              .trajectory_execution(file_path=str(config / 'moveit_controllers.yaml'))
              .planning_pipelines(pipelines=['ompl']).to_moveit_configs().to_dict())
    legacy = Path(get_package_share_directory('mycobot_mtc_pick_place_demo')) / 'config/mtc_node_params.yaml'
    params = yaml.safe_load(legacy.read_text())['mtc_node']['ros__parameters']
    params.update(moveit)
    params.update(execute=False, use_sim_time=False, end_effector='gripper', scene_source='known',
                  service_timeout=1.0, object_pose=[0.22, 0.12, 0.175, 0., 0., 0.],
                  support_surface_name='support_surface', support_surface_reference_frame='world',
                  support_surface_dimensions=[0.35, 0.35, 0.01],
                  support_surface_pose=[0.25, 0.15, -0.005, 0., 0., 0.])
    executable = Path(get_package_prefix('robot_task_core')) / 'lib/robot_task_core/mtc_node'
    rclpy.init()
    node = rclpy.create_node('task_failure_test')
    count = []

    def fail(request, response):
        count.append(request.target_shape)
        response.success = False
        return response

    service = node.create_service(GetPlanningScene, '/test_failed_scene', fail)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    cases = [
        ('invalid source', {'scene_source': 'invalid'}, 'scene_source must be known or perception'),
        ('invalid group', {'arm_group_name': 'missing_arm'}, "arm_group_name='missing_arm'"),
        ('invalid frame', {'gripper_frame': 'missing_link'}, "gripper_frame='missing_link'"),
        ('invalid pose', {'object_pose': [0., 0.]}, 'object_pose requires'),
        ('invalid dimensions', {'object_dimensions': [0.35, -1.]}, 'object_dimensions requires'),
        ('invalid support ID', {'support_surface_name': params['object_name']}, 'support_surface_name must'),
        ('known without perception', {}, 'known: built target'),
        ('perception timeout', {'scene_source': 'perception', 'scene_service_name': '/missing_scene'}, 'Service availability timeout'),
        ('perception failure', {'scene_source': 'perception', 'scene_service_name': '/test_failed_scene'}, 'Perception returned success=false'),
    ]
    try:
        with tempfile.TemporaryDirectory(prefix='task_safety_') as directory:
            for label, changes, expected in cases:
                values = copy.deepcopy(params)
                values.update(changes)
                path = Path(directory) / 'parameters.yaml'
                path.write_text(yaml.safe_dump({'mtc_node': {'ros__parameters': values}}))
                with tempfile.TemporaryFile(mode='w+') as output:
                    process = subprocess.Popen([str(executable), '--ros-args', '--params-file', str(path)],
                                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGINT)
                        process.wait(timeout=5)
                        raise AssertionError(label + ' hung')
                    output.seek(0)
                    log = output.read()
                    assert process.returncode == 1, (label, process.returncode, log[-4000:])
                    assert expected in log, (label, log[-5000:])
                    assert 'Task planning succeeded' not in log
                    if label == 'known without perception':
                        assert '[Perception] Waiting' not in log
                        assert '[MoveIt] Planning scene services unavailable' in log
                    if label == 'perception failure':
                        assert count, 'Generic perception service was never called'
                    print('Passed:', label)
    finally:
        executor.shutdown()
        spinner.join(timeout=5)
        node.destroy_service(service)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
