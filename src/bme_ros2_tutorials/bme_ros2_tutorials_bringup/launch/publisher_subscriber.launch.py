#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # 1. C++ 發布者（使用預設的 /topic）
    cpp_publisher_node = Node(
        package="bme_ros2_tutorials_cpp",
        executable="publisher_cpp",
        name="my_cpp_publisher",
    )

    # 2. Python 發布者（透過 remap 把 /topic 改成 /another_topic）
    py_publisher_node = Node(
        package="bme_ros2_tutorials_py",
        executable="py_publisher",
        name="my_py_publisher",
        remappings=[
            ("topic", "another_topic")
        ]
    )

    # 3. Python 訂閱者（監聽預設的 /topic，接收 cpp_publisher 的訊息）
    py_subscriber_node = Node(
        package="bme_ros2_tutorials_py",
        executable="py_subscriber",
        name="my_py_subscriber",
    )

    # 4. C++ 訂閱者（透過 remap 監聽 /another_topic，接收 py_publisher 的訊息）
    cpp_subscriber_node1 = Node(
        package="bme_ros2_tutorials_cpp",
        executable="subscriber_cpp",
        name="my_cpp_subscriber1",
        remappings=[
            ("topic", "another_topic")
        ]
    )

    # 將四個節點加入啟動清單
    ld.add_action(cpp_publisher_node)
    ld.add_action(py_publisher_node)
    ld.add_action(py_subscriber_node)
    ld.add_action(cpp_subscriber_node1)
    
    return ld