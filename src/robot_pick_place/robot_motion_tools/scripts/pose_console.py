#!/usr/bin/env python3

import argparse
import math
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default='kinova_gen3_lite')
    args = parser.parse_args()

    print('只規劃模式，不執行手臂動作。')
    print('輸入：X Y Z Roll Pitch Yaw')
    print('位置單位：公尺；角度單位：度。')
    print('輸入 quit 離開。')

    while True:
        try:
            text = input('目標 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n結束。')
            return

        if text.lower() in ('quit', 'exit'):
            return

        fields = text.split()

        if len(fields) != 6:
            print('請輸入六個數字，例如：0.30 0.10 0.40 0 0 10')
            continue

        try:
            values = [float(field) for field in fields]
        except ValueError:
            print('輸入必須全部是數字。')
            continue

        if not all(math.isfinite(value) for value in values):
            print('不可輸入 NaN 或無限大。')
            continue

        names = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
        command = [
            'ros2', 'launch',
            'robot_motion_tools', 'move_to_pose.launch.py',
            f'robot:={args.robot}',
        ]
        command.extend(
            f'{name}:={value}'
            for name, value in zip(names, values)
        )

        try:
            result = subprocess.run(command, check=False)
        except KeyboardInterrupt:
            print('\n已中斷，結束終端介面。')
            return
        except OSError as error:
            print(f'無法啟動規劃程式：{error}')
            continue

        if result.returncode != 0:
            print('本次未正常完成，請查看上方錯誤訊息。')

        print('本次規劃程序已結束，等待下一個目標。')


if __name__ == '__main__':
    main()