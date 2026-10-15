#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import SetParametersResult

class MyPublisherNode(Node):
    def __init__(self):
        super().__init__("python_publisher_with_parameter")
        
        # 1. 宣告參數
        self.declare_parameter("published_text", "MOGI")
        self.declare_parameter("timer_period", 1.0) # 預設頻率為 1.0 秒發布一次
        
        # 2. 讀取初始頻率並建立計時器
        self.timer_period = self.get_parameter('timer_period').value
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # 3. 關鍵：裝設「參數異動監聽器」，只要有人改參數，就會觸發 self.update_timer_period_callback
        self.add_on_set_parameters_callback(self.update_timer_period_callback)
        
        self.i = 0
        self.msg = String()
        self.get_logger().info("Publisher with parameter has been started.")

    def timer_callback(self):
        text_ = self.get_parameter("published_text").value
        self.msg.data = f"{text_}: {self.i}"
        self.i += 1
        self.get_logger().info(f'Publishing: "{self.msg.data}"')
        self.publisher_.publish(self.msg)

    def update_timer_period_callback(self, params):
        """ 當參數被外部修改時，會自動進到這裡來 """
        result = SetParametersResult(successful=True)
        for param in params:
            if param.name == 'timer_period' and param.type_ == rclpy.Parameter.Type.DOUBLE:
                new_period = param.value
                self.get_logger().info(f'Updating timer period to {new_period} seconds')
                
                # 4. 砍掉重練：取消舊鬧鐘，換上新鬧鐘！
                self.timer.cancel()
                self.timer = self.create_timer(new_period, self.timer_callback)
                return result
        return result

def main(args=None):
    rclpy.init(args=args)
    node = MyPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()