"""Start the existing AI demo, then submit one text request at a time."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid

import psutil
import rclpy
from robot_task_interfaces.srv import RunPickPlace


def stop_demo(process, partition):
    for sig, seconds in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        if process.poll() is not None:
            break
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=seconds)
        except ProcessLookupError:
            break
        except subprocess.TimeoutExpired:
            continue
    # Gazebo can leave reparented rendering workers. Only touch this launch's partition.
    owned = []
    for item in psutil.process_iter():
        try:
            if item.environ().get('GZ_PARTITION') == partition:
                owned.append(item)
                item.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(owned, timeout=3)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=2)


def main():
    parser = argparse.ArgumentParser(description='AI text pick-and-place console; starts Gazebo and MoveIt once')
    parser.add_argument('--robot', default='kinova_gen3_lite')
    parser.add_argument('--execute', action='store_true', help='Execute each task; default only plans')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', default='yoloe-26n-seg.pt')
    parser.add_argument('--confidence', type=float, default=0.05)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-rviz', action='store_true')
    parser.add_argument('--ready-timeout', type=float, default=150.0)
    parser.add_argument('--log', help='Gazebo/MoveIt/perception log file (default: unique file in /tmp)')
    args = parser.parse_args()
    if args.ready_timeout <= 0 or not 0 < args.confidence <= 1:
        parser.error('ready-timeout must be positive; confidence must be in (0, 1]')
    if args.log:
        path = Path(args.log).expanduser().resolve()
    else:
        descriptor, name = tempfile.mkstemp(prefix='robot-pick-console-', suffix='.log')
        os.close(descriptor)
        path = Path(name)
    partition = 'robot_pick_console_' + uuid.uuid4().hex[:12]
    env = dict(os.environ, GZ_PARTITION=partition)
    command = ['ros2', 'launch', 'robot_bringup', 'demo.launch.py', f'robot:={args.robot}',
               'scene_source:=perception', 'perception_method:=ai', 'interactive:=true',
               f'execute:={str(args.execute).lower()}', f'ai_device:={args.device}', f'ai_model:={args.model}',
               f'ai_confidence:={args.confidence}', f'headless:={str(args.headless).lower()}',
               f'use_rviz:={str(not args.no_rviz).lower()}', f'readiness_timeout:={args.ready_timeout}']
    process = None
    node = None
    print(f'AI 模式：{args.robot}，execute={str(args.execute).lower()}', flush=True)
    print(f'啟動中；詳細日誌：{path}', flush=True)
    try:
        with path.open('w') as output:
            process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        rclpy.init(args=[])
        node = rclpy.create_node('robot_pick_console')
        client = node.create_client(RunPickPlace, '/pick_place/run')
        deadline = time.monotonic() + args.ready_timeout
        while not client.wait_for_service(timeout_sec=0.2):
            if process.poll() is not None:
                raise RuntimeError(f'啟動失敗，請查看 {path}')
            if time.monotonic() >= deadline:
                raise RuntimeError(f'等待機器人就緒逾時，請查看 {path}')
        print('就緒。請輸入英文物品名稱，例如 drink can、green plastic bottle、tennis ball；輸入 quit 離開。', flush=True)
        while rclpy.ok():
            try:
                label = input('要夾取的物品 > ').strip()
            except EOFError:
                break
            if label.lower() in ('quit', 'exit'):
                break
            if not label:
                continue
            if process.poll() is not None or not client.service_is_ready():
                raise RuntimeError(f'任務服務已停止，請查看 {path}')
            print(f'開始處理：{label}；擷取新 RGB／深度 → 辨識 → 規劃' +
                  (' → 執行' if args.execute else '（僅規劃）'), flush=True)
            future = client.call_async(RunPickPlace.Request(target_label=label))
            while rclpy.ok() and not future.done():
                rclpy.spin_once(node, timeout_sec=0.2)
                if process.poll() is not None or not client.service_is_ready():
                    raise RuntimeError(f'任務程序已停止，請查看 {path}')
            result = future.result()
            print(('完成：' if result.success else '未完成：') + result.message, flush=True)
            print('等待下一個物品。', flush=True)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'錯誤：{error}', flush=True)
        return 1
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if process:
            stop_demo(process, partition)
    return 0
