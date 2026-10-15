#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger

class MultiThreadDemo(Node):
    def __init__(self):
        super().__init__('multithread_demo')
        
        # 1. 宣告兩組獨立的群組（員工 A 與員工 B）
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()

        # 2. 建立計時器（每 0.5 秒觸發一次）
        self.timer = self.create_timer(
            0.5, self.timer_callback, callback_group=self.timer_group
        )

        # 3. 建立服務伺服器（接收觸發請求）
        self.srv = self.create_service(
            Trigger, 'heavy_work', self.service_callback, callback_group=self.service_group
        )
        self.get_logger().info("多執行緒節點已啟動！")

    def timer_callback(self):
        self.get_logger().info('>>> 鬧鐘響起：計時器正常運作中 (0.5秒)...')

    def service_callback(self, request, response):
        self.get_logger().info('<<< 收到服務請求！開始模擬耗時 3 秒的大餐...')
        time.sleep(3.0)  # 故意卡住 3 秒
        response.success = True
        response.message = "大餐做好了！"
        self.get_logger().info('<<< 服務處理完畢！')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = MultiThreadDemo()
    
    # 4. 關鍵：使用雙執行緒執行器（開 2 個員工）
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()