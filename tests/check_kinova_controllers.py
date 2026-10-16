"""Against a running Lite simulation, verify real controller goals and joint feedback.

This moves the otherwise idle arm and gripper. Run only in the dedicated test simulation.
"""
import time
import rclpy
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


def main():
    rclpy.init()
    node = rclpy.create_node('kinova_controller_check')
    positions = {}
    node.create_subscription(JointState, '/joint_states',
                             lambda msg: positions.update(zip(msg.name, msg.position)), 10)
    gripper = ActionClient(node, GripperCommand, '/gen3_lite_2f_gripper_controller/gripper_cmd')
    arm = ActionClient(node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

    def result(future):
        rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
        assert future.done(), 'Controller action timeout'
        return future.result()

    def send(client, goal):
        handle = result(client.send_goal_async(goal))
        assert handle.accepted, 'Controller rejected goal'
        response = result(handle.get_result_async())
        assert response.status == 4, response
        return response.result

    def feedback(joint, expected):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if abs(positions.get(joint, float('inf')) - expected) < 0.025:
                return
        raise AssertionError((joint, expected, positions))

    try:
        assert gripper.wait_for_server(timeout_sec=30.0), 'Gripper action unavailable'
        assert arm.wait_for_server(timeout_sec=30.0), 'Arm action unavailable'
        for position in [0.35, 0.0]:
            goal = GripperCommand.Goal()
            goal.command.position = position
            goal.command.max_effort = 10.0
            assert send(gripper, goal).reached_goal
            feedback('right_finger_bottom_joint', position)
        print('Passed: Gen3 Lite 2F GripperCommand open/close and master joint feedback')
        for position in [0.1, 0.0]:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = [f'joint_{i}' for i in range(1, 7)]
            point = JointTrajectoryPoint()
            point.positions = [position, 0.0, 0.0, 0.0, 0.0, 0.0]
            point.time_from_start.sec = 2
            goal.trajectory.points = [point]
            assert send(arm, goal).error_code == FollowJointTrajectory.Result.SUCCESSFUL
            feedback('joint_1', position)
        print('Passed: six-joint FollowJointTrajectory and arm joint feedback; returned home')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
