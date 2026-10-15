#!/usr/bin/env python3
from bme_ros2_tutorials_interfaces.srv import CustomCalc  # 匯入自訂服務
import rclpy
from rclpy.node import Node

class MyService(Node):
    def __init__(self):
        super().__init__('my_service')
        # 建立 Service Server，通道名稱為 'custom_calc'
        self.srv = self.create_service(CustomCalc, 'custom_calc', self.custom_calc_callback)
        self.get_logger().info("Service Server has been started.")

    def custom_calc_callback(self, request, response):
        # 處理請求：將 a 和 b 相加存入 result
        response.result = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main():
    rclpy.init()
    my_service_server = MyService()
    rclpy.spin(my_service_server)
    rclpy.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()