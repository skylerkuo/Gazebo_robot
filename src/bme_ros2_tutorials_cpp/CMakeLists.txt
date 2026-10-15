cmake_minimum_required(VERSION 3.8)
project(bme_ros2_tutorials_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 尋找相依的 ROS 2 套件
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# 2. 定義要編譯的 C++ 執行檔
add_executable(publisher_cpp src/publisher.cpp)
add_executable(subscriber_cpp src/subscriber.cpp)
ament_target_dependencies(publisher_cpp rclcpp std_msgs)
ament_target_dependencies(subscriber_cpp rclcpp std_msgs)

# 3. 設定安裝路徑
install(TARGETS
  publisher_cpp
  subscriber_cpp
  DESTINATION lib/${PROJECT_NAME})

ament_package()