#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from bme_ros2_tutorials_interfaces.action import CountUntil

class CountUntilClient(Node):
    def __init__(self):
        super().__init__('count_until_client')
        self._action_client = ActionClient(self, CountUntil, 'count_until')

    def send_goal(self, target_number):
        goal_msg = CountUntil.Goal()
        goal_msg.target_number = target_number

        self.get_logger().info('正在尋找 Action 伺服器...')
        self._action_client.wait_for_server()

        self.get_logger().info(f'發送目標：請幫我數到 {target_number}！')
        
        # 非同步發送目標，並指定接收 Feedback 的回呼函式
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('伺服器拒絕了這個目標 :(')
            return

        self.get_logger().info('伺服器已接受目標，等待結果中...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        # 隨時接收伺服器傳來的即時進度
        feedback = feedback_msg.feedback
        self.get_logger().info(f'【即時進度回報】收到伺服器回報：現在數到 {feedback.current_number}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'【最終結果】任務結束！最終數到：{result.reached_number}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    client = CountUntilClient()
    client.send_goal(5)  # 派發任務：請幫我數到 5！
    rclpy.spin(client)

if __name__ == '__main__':
    main()