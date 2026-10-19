"""Bounded end-to-end demo check. Only terminates the launch process group it creates."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil


def stop(process):
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        os.killpg(process.pid, sig)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def cleanup_partition(partition):
    # All children inherit this test's unique partition, including reparented gz workers.
    owned = []
    for process in psutil.process_iter():
        try:
            if process.environ().get('GZ_PARTITION') == partition:
                owned.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for process in owned:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)


def snapshot(env, path, source, method='geometry'):
    commands = [
        ['ros2', 'node', 'list', '--no-daemon', '--spin-time', '3'],
        ['ros2', 'control', 'list_controllers'],
        ['ros2', 'topic', 'list', '--no-daemon', '-t'],
        ['ros2', 'service', 'list', '--no-daemon', '-t'],
        ['ros2', 'action', 'list', '-t'],
    ]
    with open(path + '.cli', 'w') as output:
        for command in commands:
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
            output.write(' '.join(command) + '\n' + result.stdout + result.stderr + '\n')
            assert result.returncode == 0, (command, result.stderr)
            if command[1:3] == ['node', 'list']:
                names = result.stdout.splitlines()
                for node in ('/move_group', '/controller_manager', '/robot_state_publisher'):
                    assert names.count(node) == 1, (node, names)
                assert ('/robot_perception_server' in names) == (source == 'perception'), names
                assert ('/robot_ai_perception_server' in names) == (source == 'perception' and method == 'ai'), names
    print('Passed: CLI controllers/topics/services/actions and unique core node counts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default='mycobot_280')
    parser.add_argument('--task-params-file', default='', help='Optional existing bringup task/scene override YAML')
    parser.add_argument('--execute', action='store_true', help='Execute the full task in this isolated Gazebo test')
    parser.add_argument('--with-gui', action='store_true', help='Test Gazebo GUI and RViz as in the normal launch')
    parser.add_argument('--scene-source', default='known', choices=['known', 'perception'])
    parser.add_argument('--perception-method', default='geometry', choices=['geometry', 'ai'])
    parser.add_argument('--ai-target-class', default='drink can')
    parser.add_argument('--ai-model', default='yoloe-26n-seg.pt')
    parser.add_argument('--ai-device', default='cpu')
    parser.add_argument('--timeout', default=150.0, type=float)
    parser.add_argument('--log', default='/tmp/robot-demo-check.log')
    arguments = parser.parse_args()
    env = dict(os.environ, ROS_HOME='/tmp/robot-refactor-home', ROS_LOG_DIR='/tmp/robot-refactor-logs', ROS_DOMAIN_ID=str(170 + os.getpid() % 40),
               GZ_PARTITION=f'robot_refactor_{os.getpid()}')
    command = ['ros2', 'launch', 'robot_bringup', 'demo.launch.py', f'robot:={arguments.robot}',
               f'scene_source:={arguments.scene_source}', f'perception_method:={arguments.perception_method}',
               f'execute:={str(arguments.execute).lower()}', 'interactive:=false',
               f'use_rviz:={str(arguments.with_gui).lower()}', f'headless:={str(not arguments.with_gui).lower()}',
               f'readiness_timeout:={min(arguments.timeout, 120.0)}']
    if arguments.task_params_file:
        command.append(f'task_params_file:={arguments.task_params_file}')
    if arguments.perception_method == 'ai':
        command.extend([f'ai_target_class:={arguments.ai_target_class}', f'ai_model:={arguments.ai_model}',
                        f'ai_device:={arguments.ai_device}'])
    with open(arguments.log, 'w') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + arguments.timeout
            while process.poll() is None and time.monotonic() < deadline:
                log = Path(arguments.log).read_text(errors='replace')
                completion = 'Task executed successfully' if arguments.execute else 'Execution skipped as per configuration'
                if 'Task planning succeeded' in log and completion in log:
                    snapshot(env, arguments.log, arguments.scene_source, arguments.perception_method)
                    print('Passed: complete demo plan and execution' if arguments.execute else
                          'Passed: complete demo plan; execute=false skipped execution')
                    return 0
                time.sleep(0.5)
            print('Failed or unavailable: demo did not complete planning/execution; inspect', arguments.log)
            return 1
        finally:
            stop(process)
            cleanup_partition(env['GZ_PARTITION'])


if __name__ == '__main__':
    raise SystemExit(main())
