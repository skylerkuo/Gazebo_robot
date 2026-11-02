#!/usr/bin/env python3
"""Record independent evidence of clocks, observations and commanded arm motion."""
import json
from pathlib import Path
import signal
import struct
import sys
import time
import zlib

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from aic_control_interfaces.msg import MotionUpdate
from aic_model_interfaces.msg import Observation
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def save_png(path, msg):
    if msg.encoding not in ("rgb8", "bgr8"):
        return
    image = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    image = image[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        image = image[:, :, ::-1]
    # Keep camera evidence small without requiring extra imaging packages.
    image = image[::4, ::4].copy()
    height, width = image.shape[:2]
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(scanlines, 1)) + chunk(b"IEND", b"")
    )


def main():
    output = Path(sys.argv[1])
    rclpy.init()
    node = rclpy.create_node("robot_insert_monitor")
    stopped = False
    started = time.monotonic()
    evidence = {
        "clock_messages": 0, "clock_first": None, "clock_last": None,
        "joint_messages": 0, "observations": 0, "pose_commands": 0,
        "joint_min": {}, "joint_max": {}, "cameras": {},
        "insertion_events": [],
    }
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    def clock(msg):
        stamp = msg.clock.sec + msg.clock.nanosec / 1e9
        evidence["clock_messages"] += 1
        if evidence["clock_first"] is None:
            evidence["clock_first"] = stamp
        evidence["clock_last"] = stamp
    def joints(msg):
        evidence["joint_messages"] += 1
        for name, value in zip(msg.name, msg.position):
            evidence["joint_min"][name] = min(evidence["joint_min"].get(name, value), value)
            evidence["joint_max"][name] = max(evidence["joint_max"].get(name, value), value)
    def command(_):
        evidence["pose_commands"] += 1
    def insertion(msg):
        evidence["insertion_events"].append({
            "simulation_time": evidence["clock_last"], "data": msg.data,
        })
    def observation(msg):
        evidence["observations"] += 1
        if evidence["observations"] != 1 and evidence["observations"] % 100:
            return
        for name in ("left", "center", "right"):
            image = getattr(msg, name + "_image")
            evidence["cameras"][name] = {"width": image.width, "height": image.height, "encoding": image.encoding}
            save_png(output / (name + "_camera.png"), image)
    subscriptions = [
        node.create_subscription(Clock, "/clock", clock, qos_profile_sensor_data),
        node.create_subscription(JointState, "/joint_states", joints, qos_profile_sensor_data),
        node.create_subscription(MotionUpdate, "/aic_controller/pose_commands", command, qos_profile_sensor_data),
        node.create_subscription(Observation, "/observations", observation, qos_profile_sensor_data),
        node.create_subscription(String, "/scoring/insertion_event", insertion, qos_profile_sensor_data),
    ]
    def write_evidence():
        evidence["wall_seconds"] = round(time.monotonic() - started, 2)
        evidence["joint_range_radians"] = {
            name: evidence["joint_max"][name] - value for name, value in evidence["joint_min"].items()
        }
        temporary = output / "evidence.json.partial"
        temporary.write_text(json.dumps(evidence, indent=2) + "\n")
        temporary.replace(output / "evidence.json")
    node.create_timer(2.0, write_evidence)
    try:
        while not stopped and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.25)
    finally:
        write_evidence()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
