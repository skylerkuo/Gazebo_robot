cmake_minimum_required(VERSION 3.8)
project(robot_task_core)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(ament_cmake REQUIRED)
set(dependencies rclcpp rclcpp_action moveit_core moveit_ros_planning_interface
  moveit_task_constructor_core moveit_task_constructor_msgs robot_task_interfaces
  moveit_msgs geometry_msgs shape_msgs sensor_msgs tf2_ros tf2_eigen tf2_geometry_msgs
  controller_manager_msgs control_msgs)
foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()
add_library(robot_task_scene_client SHARED src/get_planning_scene_client.cpp)
target_include_directories(robot_task_scene_client PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
ament_target_dependencies(robot_task_scene_client ${dependencies})
add_executable(mtc_node src/mtc_node.cpp)
ament_target_dependencies(mtc_node ${dependencies})
target_link_libraries(mtc_node robot_task_scene_client)
install(TARGETS robot_task_scene_client mtc_node
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
ament_package()
