"""Real Gazebo/YOLOE/MTC checks for command gating and repeated requests."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time

import rclpy
from robot_task_interfaces.srv import RunPickPlace
from run_demo_check import stop, cleanup_partition, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--log', default='/tmp/kinova-ai-commands.log')
    args = parser.parse_args()
    env = dict(os.environ, ROS_DOMAIN_ID=str(170 + os.getpid() % 40),
               ROS_LOG_DIR='/tmp/robot-ai-commands-logs', GZ_PARTITION=f'ai_commands_{os.getpid()}')
    os.environ['ROS_DOMAIN_ID'] = env['ROS_DOMAIN_ID']
    path = Path(args.log)
    command = ['ros2', 'launch', 'robot_bringup', 'demo.launch.py', 'robot:=kinova_gen3_lite',
               'perception_method:=ai', 'interactive:=true', 'use_rviz:=false', 'headless:=true',
               f'execute:={str(args.execute).lower()}']
    with path.open('w') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    rclpy.init()
    node = rclpy.create_node('ai_commands_check')
    client = node.create_client(RunPickPlace, '/pick_place/run')

    def until(predicate, seconds=100):
        deadline = time.monotonic() + seconds
        while not predicate() and time.monotonic() < deadline:
            assert process.poll() is None, path.read_text()[-2000:]
            rclpy.spin_once(node, timeout_sec=0.1)
        assert predicate(), f'Timeout: inspect {path}'

    def result(future, seconds=150):
        until(future.done, seconds)
        return future.result()

    def assert_idle():
        baseline = path.read_text().count('[AI] Request target=')
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert path.read_text().count('[AI] Request target=') == baseline
        for topic in ('/camera_head/color/image_raw', '/camera_head/depth/color/points'):
            names = [info.node_name for info in node.get_subscriptions_info_by_topic(topic)]
            assert 'robot_perception_server' not in names, names

    try:
        until(client.service_is_ready, 150)
        assert_idle()
        assert '[Capture]' not in path.read_text()
        assert 'Starting the pick and place task' not in path.read_text()
        empty = result(client.call_async(RunPickPlace.Request(target_label=' ')))
        assert not empty.success and 'empty' in empty.message
        assert '[AI] Request' not in path.read_text()
        print('Passed: idle/blank input does no image capture, YOLOE, PCL, or MTC work', flush=True)

        first = client.call_async(RunPickPlace.Request(target_label='drink can'))
        until(lambda: "[Command] Starting target='drink can'" in path.read_text())
        busy = result(client.call_async(RunPickPlace.Request(target_label='tennis ball')), 3)
        assert not busy.success and 'busy' in busy.message, busy
        first_result = result(first)
        assert first_result.success and first_result.executed == args.execute, first_result
        print('Passed: one command completes; overlapping command rejected', flush=True)
        assert_idle()

        # Unknown target must fail, then the same process must accept a new request.
        missing = result(client.call_async(RunPickPlace.Request(target_label='purple helicopter')))
        assert not missing.success, missing
        assert_idle()
        # Executed objects are collected outside the pickup ROI; pick another item.
        second_label = 'green plastic bottle' if args.execute else 'drink can'
        second = result(client.call_async(RunPickPlace.Request(target_label=second_label)))
        assert second.success and second.executed == args.execute, second
        assert_idle()
        log = path.read_text()
        assert log.count('[AI] Request target=') == 3
        captures = re.findall(r'\[Capture\] Fresh RGB/depth after request: start=(\d+) cloud=(\d+) image=(\d+)', log)
        assert len(captures) == 3, captures
        assert all(int(cloud) > int(start) and int(image) > int(start) for start, cloud, image in captures)
        assert int(captures[2][0]) > int(captures[0][0])
        assert log.count('Task planning succeeded') == 2
        snapshot(env, args.log, 'perception', 'ai')
        print('Passed: failed target then second full task; all captures newer than their request', flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        stop(process)
        cleanup_partition(env['GZ_PARTITION'])


if __name__ == '__main__':
    main()
