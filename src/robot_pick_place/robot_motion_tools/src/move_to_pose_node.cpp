#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.hpp>

// 外部有傳入就讀取，沒有才宣告預設值。
template<typename T>
T read_parameter(
  const rclcpp::Node::SharedPtr & node,
  const std::string & name,
  const T & default_value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter<T>(name, default_value);
  }
  return node->get_parameter(name).get_value<T>();
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
    "move_to_pose",
    rclcpp::NodeOptions()
      .automatically_declare_parameters_from_overrides(true));

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  int result = 1;

  try {
    const auto arm_group =
      read_parameter<std::string>(node, "arm_group", "");
    const auto reference_frame =
      read_parameter<std::string>(node, "reference_frame", "");
    const auto end_effector_link =
      read_parameter<std::string>(node, "end_effector_link", "");
    const auto pipeline =
      read_parameter<std::string>(node, "planning_pipeline", "");
    const auto planner_id =
      read_parameter<std::string>(node, "planner_id", "");
    const bool execute =
        read_parameter<bool>(node, "execute", false);

    // 只是初期練習目標，不保證可達；下一段由啟動設定傳入。
    const double x = read_parameter<double>(node, "target_x", 0.30);
    const double y = read_parameter<double>(node, "target_y", 0.10);
    const double z = read_parameter<double>(node, "target_z", 0.40);
    const double roll_deg =
        read_parameter<double>(node, "roll_deg", 0.0);
    const double pitch_deg =
        read_parameter<double>(node, "pitch_deg", 0.0);
    const double yaw_deg =
        read_parameter<double>(node, "yaw_deg", 0.0);

    if (!std::isfinite(roll_deg) ||
        !std::isfinite(pitch_deg) ||
        !std::isfinite(yaw_deg))
    {
        throw std::runtime_error("RPY angles must be finite numbers.");
    }

    if (arm_group.empty() || reference_frame.empty() ||
        end_effector_link.empty() || pipeline.empty() ||
        planner_id.empty())
    {
      throw std::runtime_error(
        "Missing group, frame, end-effector or planner parameters.");
    }

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      throw std::runtime_error("Target XYZ must be finite numbers.");
    }

    using moveit::planning_interface::MoveGroupInterface;
    MoveGroupInterface move_group(node, arm_group);

    move_group.setPoseReferenceFrame(reference_frame);

    if (!move_group.setEndEffectorLink(end_effector_link)) {
      throw std::runtime_error("Invalid end-effector link.");
    }

    move_group.setPlanningPipelineId(pipeline);
    move_group.setPlannerId(planner_id);
    move_group.setPlanningTime(5.0);
    move_group.setMaxVelocityScalingFactor(0.1);
    move_group.setMaxAccelerationScalingFactor(0.1);

    const auto current_state = move_group.getCurrentState(10.0);
    if (!current_state) {
      throw std::runtime_error("Could not obtain current robot state.");
    }
    move_group.setStartState(*current_state);

    geometry_msgs::msg::PoseStamped target;
    target.header.frame_id = reference_frame;
    target.header.stamp = node->now();
    target.pose.position.x = x;
    target.pose.position.y = y;
    target.pose.position.z = z;

    const double deg_to_rad = std::acos(-1.0) / 180.0;

    tf2::Quaternion orientation;
    orientation.setRPY(
        roll_deg * deg_to_rad,
        pitch_deg * deg_to_rad,
        yaw_deg * deg_to_rad);
    orientation.normalize();

    target.pose.orientation.x = orientation.x();
    target.pose.orientation.y = orientation.y();
    target.pose.orientation.z = orientation.z();
    target.pose.orientation.w = orientation.w();

    RCLCPP_INFO(
        node->get_logger(),
        "Target RPY degrees: roll=%.1f pitch=%.1f yaw=%.1f",
        roll_deg, pitch_deg, yaw_deg);

    if (!move_group.setPoseTarget(target, end_effector_link)) {
      throw std::runtime_error("Could not set pose target.");
    }

    RCLCPP_INFO(
      node->get_logger(),
      "Target: frame=%s link=%s XYZ=(%.3f, %.3f, %.3f)",
      reference_frame.c_str(), end_effector_link.c_str(), x, y, z);

    MoveGroupInterface::Plan plan;
    const auto code = move_group.plan(plan);

    if (code == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(
        node->get_logger(),
        "Planning success. Planning time: %.3f s.",
        plan.planning_time);

      if (!execute) {
        RCLCPP_INFO(node->get_logger(), "Plan only. No execution requested.");
        result = 0;
      } else {
        RCLCPP_INFO(node->get_logger(), "Executing planned trajectory.");

        const auto execution_code = move_group.execute(plan);

        if (execution_code == moveit::core::MoveItErrorCode::SUCCESS) {
          RCLCPP_INFO(node->get_logger(), "Execution reported success.");
          result = 0;
        } else {
          RCLCPP_ERROR(
            node->get_logger(),
            "Execution failed. MoveIt error code: %d",
            execution_code.val);
        }
      }
    } else {
      RCLCPP_ERROR(
        node->get_logger(), "Planning failed. MoveIt error code: %d",
        code.val);
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "%s", error.what());
  }

  executor.cancel();
  spinner.join();
  rclcpp::shutdown();
  return result;
}
