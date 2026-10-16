#!/usr/bin/env python3
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from bme_ros2_tutorials_interfaces.action import CountUntil

class CountUntilServer(Node):
    def __init__(self):
        super().__init__('count_until_server')
        
        # 建立 Action 伺服器，綁定自訂型態與名稱 'count_until'
        self._action_server = ActionServer(
            self,
            CountUntil,
            'count_until',
            self.execute_callback
        )
        self.get_logger().info("Action Server 已啟動，等待任務中...")

    def execute_callback(self, goal_handle):
        self.get_logger().info(f'收到目標！準備數到：{goal_handle.request.target_number}')
        
        feedback_msg = CountUntil.Feedback()
        
        # 開始執行長時間任務（模擬每秒數一個數字）
        for i in range(1, goal_handle.request.target_number + 1):
            
            # 檢查是否有被客戶端取消請求
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('任務已被客戶端取消！')
                return CountUntil.Result(reached_number=i-1)
            
            # 填入並發送即時回饋（Feedback）
            feedback_msg.current_number = i
            self.get_logger().info(f'回報進度：目前數到 {i}')
            goal_handle.publish_feedback(feedback_msg)
            
            time.sleep(1.0)  # 模擬每秒做一件事
        
        # 任務圓滿成功
        goal_handle.succeed()
        
        result = CountUntil.Result()
        result.reached_number = goal_handle.request.target_number
        self.get_logger().info('任務圓滿達成！')
        return result

def main(args=None):
    rclpy.init(args=args)
    node = CountUntilServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()