#!/usr/bin/env python3
import rclpy
from std_msgs.msg import String

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('python_subscriber')

    def subscriber_callback(msg):
        node.get_logger().info(f"I heard: {msg.data}")

    subscriber = node.create_subscription(String, 'topic', subscriber_callback, 10) 
    node.get_logger().info("Subsciber has been started.")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()