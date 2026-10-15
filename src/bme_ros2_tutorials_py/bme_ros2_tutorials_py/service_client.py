#!/usr/bin/env python3
import sys
from bme_ros2_tutorials_interfaces.srv import CustomCalc  # 匯入自訂服務
import rclpy
from rclpy.node import Node

class MyServiceClientAsync(Node):
    def __init__(self):
        super().__init__('my_service_client_async')
        # 建立 Service Client
        self.cli = self.create_client(CustomCalc, 'custom_calc')
        
        # 確保伺服器已經上線，沒上線就每秒印出等待訊息
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
            
        self.req = CustomCalc.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        # 非同步發送請求
        return self.cli.call_async(self.req)

def main():
    rclpy.init()
    my_service_client = MyServiceClientAsync()
    
    # 從命令列讀取兩個數字作為參數 (例如: ros2 run ... client 3 4)
    a_val = int(sys.argv[1])
    b_val = int(sys.argv[2])
    
    future = my_service_client.send_request(a_val, b_val)
    
    # 持續轉圈直到收到回應
    rclpy.spin_until_future_complete(my_service_client, future)
    response = future.result()
    
    my_service_client.get_logger().info(
        'Result of custom_calc: for %d + %d = %d' % (a_val, b_val, response.result)
    )

    my_service_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()