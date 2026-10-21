/**
 * @file mtc_node.cpp
 * @brief Implementation of a MoveIt Task Constructor (MTC) node for a pick and place task.
 *
 * This program implements a pick and place task using MoveIt Task Constructor (MTC).
 * It creates a planning scene, generates a series of motion stages, and executes them
 * to pick up an object from one location and place it in another.
 *
 * The node performs the following main steps:
 * 1. Set up the planning scene with collision objects
 * 2. Create an MTC task with stages for picking and placing
 * 3. Plan the task
 * 4. Optionally execute the planned task
 *
 * Input Parameters:
 * - Robot configuration (arm_group_name, gripper_group_name, etc.)
 * - Object parameters (name, dimensions, pose)
 * - Motion planning parameters (approach distances, timeouts, etc.)
 * - Execution flag (whether to execute the planned task)
 *
 * Output:
 * - Planned trajectory for the pick and place task
 * - Execution of the planned trajectory (if execution flag is set)
 * - Visualization markers for RViz
 *
 * Services Used:
 * - GetPlanningScene: Retrieves the current planning scene
 *
 * Topics Published:
 * - Various topics for trajectory execution and visualization (specific topics depend on MoveIt configuration)
 *
 * @author Addison Sears-Collins
 * @date December 20, 2024
 */

// Include necessary ROS 2 and MoveIt headers
#include <rclcpp/rclcpp.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>

// Other utilities
#include <algorithm>
#include <cmath>
#include <future>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <controller_manager_msgs/srv/list_controllers.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <type_traits>
#include <numeric>
#include <string>
#include <vector>
#include <rclcpp/serialization.hpp>
#include <moveit/task_constructor/introspection.h>

// Handling the GetPlanningScene service
#include "robot_task_core/get_planning_scene_client.h"
#include <robot_task_interfaces/srv/run_pick_place.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>

// Conditional includes for tf2 geometry messages and Eigen
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
#include <tf2_eigen/tf2_eigen.hpp>
#else
#include <tf2_eigen/tf2_eigen.h>
#endif

namespace {
/**
 * @brief Transform a vector of numbers into a 3D position and orientation.
 * @param values Vector containing position and orientation values.
 * @return Eigen::Isometry3d representing the transformation.
 */
Eigen::Isometry3d vectorToEigen(const std::vector<double>& values) {
  Eigen::Quaterniond rotation;
  if (values.size() == 7) {
    rotation = Eigen::Quaterniond(values[6], values[3], values[4], values[5]).normalized();
  } else {
    // ROS fixed-axis roll/pitch/yaw: Rz(yaw) * Ry(pitch) * Rx(roll).
    rotation = Eigen::AngleAxisd(values[5], Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(values[4], Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(values[3], Eigen::Vector3d::UnitX());
  }
  return Eigen::Translation3d(values[0], values[1], values[2]) * rotation;
}

/**
 * @brief Convert a vector of numbers to a geometry_msgs::msg::Pose.
 * @param values Vector containing position and orientation values.
 * @return geometry_msgs::msg::Pose representing the pose.
 */
geometry_msgs::msg::Pose vectorToPose(const std::vector<double>& values) {
  return tf2::toMsg(vectorToEigen(values));
};
// Mimic joints can sit on sibling branches outside the SRDF group. Include their
// moving collision geometry when allowing finger contact with the grasped object.
std::vector<std::string> gripperTouchLinks(const moveit::core::RobotModelConstPtr& model,
                                          const std::string& group_name)
{
  const auto* group = model->getJointModelGroup(group_name);
  const auto& updated = group->getUpdatedLinkModelsWithGeometryNames();
  std::set<std::string> names(updated.begin(), updated.end());
  for (const auto* mimic : model->getMimicJointModels()) {
    const auto* driver = mimic;
    while (driver->getMimic()) driver = driver->getMimic();
    if (!group->hasJointModel(driver->getName())) continue;
    const auto* child = mimic->getChildLinkModel();
    if (!child->getShapes().empty()) names.insert(child->getName());
    for (const auto* link : mimic->getDescendantLinkModels())
      if (!link->getShapes().empty()) names.insert(link->getName());
  }
  return {names.begin(), names.end()};
}
}  // namespace

// Namespace alias for MoveIt Task Constructor
namespace mtc = moveit::task_constructor;

/**
 * @brief Class representing the MTC Task Node.
 */
class MTCTaskNode : public rclcpp::Node
{
public:
  MTCTaskNode(const rclcpp::NodeOptions& options);

  void validateConfiguration();
  void doTask();
  void setupPlanningScene(const std::string& target_label = "");
  void startCommandService();

private:
  mtc::Task task_;
  mtc::Task createTask();
  void setupSceneFromKnownObjects();
  void setupSceneFromPerception(const std::string& target_label);
  void validateFrame(const std::string& frame, const std::string& parameter);
  void checkExecutionControllers();
  void appendCollectionBin();
  void saveDemonstration(const std::string& state, int code = 0);
  int64_t execution_start_ns_ = 0;
  int64_t execution_end_ns_ = 0;

  void updateObjectParameters(const moveit_msgs::msg::CollisionObject& collision_object);

  // Variables for calling the GetPlanningScene service
  std::shared_ptr<GetPlanningSceneClient> planning_scene_client;
  moveit_msgs::msg::PlanningSceneWorld scene_world_;
  sensor_msgs::msg::PointCloud2 full_cloud_;
  sensor_msgs::msg::Image rgb_image_;
  std::string target_object_id_;
  std::string support_surface_id_;
  bool scene_ready_ = false;
  std::set<std::string> managed_scene_ids_;
  std::mutex command_mutex_;
  rclcpp::CallbackGroup::SharedPtr command_group_;
  rclcpp::Service<robot_task_interfaces::srv::RunPickPlace>::SharedPtr command_service_;
};

/**
 * @brief Constructor for the MTCTaskNode class.
 * @param options Node options for configuration.
 */
MTCTaskNode::MTCTaskNode(const rclcpp::NodeOptions& options)
  : Node("mtc_node", options)
{
  auto declare_parameter = [this](const std::string& name, const auto& default_value, const std::string& description = "") {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = description;

    if (!this->has_parameter(name)) {
      this->declare_parameter(name, default_value, descriptor);
    }
  };

  // General parameters
  declare_parameter("scene_source", "known", "known or perception");
  declare_parameter("scene_service_name", "/perception/get_planning_scene", "Generic scene service");
  declare_parameter("service_timeout", 30.0, "Service availability and response timeout in seconds");
  declare_parameter("controller_manager", "/controller_manager", "Controller manager namespace");
  declare_parameter("gripper_strategy", "named_pose", "Supported strategy: named_pose; other actions require an adapter");
  declare_parameter("end_effector", "", "SRDF end effector name (distinct from gripper group)");
  declare_parameter("planning_pipeline", "ompl", "MoveIt planning pipeline ID");
  declare_parameter("planner_id", "RRTConnectkConfigDefault", "Planner configuration ID");
  declare_parameter("support_surface_name", "", "Known scene support object ID");
  declare_parameter("support_surface_reference_frame", "", "Known support frame");
  declare_parameter("support_surface_dimensions", std::vector<double>{}, "Support BOX [x,y,z]");
  declare_parameter("support_surface_pose", std::vector<double>{}, "Support center pose");
  declare_parameter("execute", false, "Whether to execute the planned task");
  declare_parameter("pick_only", false, "Finish after grasp and lift, keeping the object attached");
  declare_parameter("demonstration_directory", "", "Optional directory for selected solution and task status");
  declare_parameter("known_obstacles_file", "", "Optional serialized PlanningSceneWorld with additional known obstacles");
  declare_parameter("interactive", false, "Wait for one RunPickPlace request per task");
  declare_parameter("command_service_name", "/pick_place/run", "Repeated task command service");
  declare_parameter("max_solutions", 25, "Maximum number of solutions to compute");

  // Controller parameters
  declare_parameter("controller_names", std::vector<std::string>{}, "Names of the controllers to use");

  // Robot configuration parameters
  declare_parameter("arm_group_name", "", "Name of the arm group in the SRDF");
  declare_parameter("gripper_group_name", "", "Name of the gripper group in the SRDF");
  declare_parameter("gripper_frame", "", "Name of the gripper frame");
  declare_parameter("gripper_open_pose", "", "Name of the gripper open pose");
  declare_parameter("gripper_close_pose", "", "Name of the gripper closed pose");
  declare_parameter("arm_home_pose", "", "Name of the arm home pose");

  // Scene frame parameters
  declare_parameter("world_frame", "", "Name of the world frame");

  // Object parameters
  declare_parameter("object_name", "", "Name of the object to be manipulated");
  declare_parameter("object_type", "cylinder", "Type of the object to be manipulated");
  declare_parameter("object_reference_frame", "", "Reference frame for the object");
  declare_parameter("object_dimensions", std::vector<double>{}, "Dimensions of the object [height, radius]");
  declare_parameter("object_pose", std::vector<double>{}, "Initial pose of the object [x, y, z, roll, pitch, yaw]");

  // Grasp and place parameters
  declare_parameter("grasp_frame_transform", std::vector<double>{}, "Transform from gripper frame to grasp frame [x, y, z, roll, pitch, yaw]");
  declare_parameter("place_pose", std::vector<double>{}, "Pose where the object should be placed [x, y, z, roll, pitch, yaw]");
  declare_parameter("drop_into_bin", false, "Release above a collection bin instead of lowering onto a surface");
  declare_parameter("bin_pose", std::vector<double>{}, "Bin bottom-center pose in world_frame");
  declare_parameter("bin_dimensions", std::vector<double>{}, "Bin outer dimensions [x,y,z]");
  declare_parameter("bin_wall_thickness", 0.01, "Bin floor/wall thickness");

  // Motion planning parameters
  declare_parameter("approach_object_min_dist", 0.0015, "Minimum approach distance to the object");
  declare_parameter("approach_object_max_dist", 0.3, "Maximum approach distance to the object");
  declare_parameter("lift_object_min_dist", 0.005, "Minimum lift distance for the object");
  declare_parameter("lift_object_max_dist", 0.3, "Maximum lift distance for the object");
  declare_parameter("lower_object_min_dist", 0.005, "Minimum distance for lowering object");
  declare_parameter("lower_object_max_dist", 0.4, "Maximum distance for lowering object");

  // Timeout parameters
  declare_parameter("move_to_pick_timeout", 10.0, "Timeout for move to pick stage (seconds)");
  declare_parameter("move_to_place_timeout", 10.0, "Timeout for move to place stage (seconds)");

  // Grasp generation parameters
  declare_parameter("grasp_pose_angle_delta", 0.1309, "Angular resolution for sampling grasp poses (radians)");
  declare_parameter("grasp_pose_max_ik_solutions", 10, "Maximum number of IK solutions for grasp pose generation");
  declare_parameter("grasp_pose_min_solution_distance", 0.8, "Minimum distance in joint-space units between IK solutions for grasp pose");

  // Place generation parameters
  declare_parameter("place_pose_max_ik_solutions", 10, "Maximum number of IK solutions for place pose generation");

  // Cartesian planner parameters
  declare_parameter("cartesian_max_velocity_scaling", 1.0, "Max velocity scaling factor for Cartesian planner");
  declare_parameter("cartesian_max_acceleration_scaling", 1.0, "Max acceleration scaling factor for Cartesian planner");
  declare_parameter("cartesian_step_size", 0.00025, "Step size for Cartesian planner");

  // Direction vector parameters
  declare_parameter("approach_object_direction_z", 1.0, "Z component of approach object direction vector");
  declare_parameter("lift_object_direction_z", 1.0, "Z component of lift object direction vector");
  declare_parameter("lower_object_direction_z", -1.0, "Z component of lower object direction vector");
  declare_parameter("retreat_direction_z", -1.0, "Z component of retreat direction vector");

  // Other parameters
  declare_parameter("place_pose_z_offset_factor", 0.5, "Factor to multiply object height for place pose Z offset");
  declare_parameter("retreat_min_distance", 0.025, "Minimum distance for retreat motion");
  declare_parameter("retreat_max_distance", 0.25, "Maximum distance for retreat motion");

  RCLCPP_INFO(this->get_logger(), "All parameters have been declared with descriptions");


}

// Validate before service waits or stage construction, so configuration errors fail early.
void MTCTaskNode::validateConfiguration()
{
  auto check_vector = [this](const std::string& name, size_t size, bool positive = false) {
    const auto values = get_parameter(name).as_double_array();
    if (values.size() != size || !std::all_of(values.begin(), values.end(),
        [positive](double v) { return std::isfinite(v) && (!positive || v > 0.0); })) {
      throw std::runtime_error("[MTC configuration] " + name + " requires " +
          std::to_string(size) + (positive ? " positive finite values" : " finite values"));
    }
  };
  const auto type = get_parameter("object_type").as_string();
  if (type != "cylinder" && type != "box")
    throw std::runtime_error("[Scene] object_type must be cylinder or box");
  check_vector("object_dimensions", type == "cylinder" ? 2 : 3, true);
  const auto source = get_parameter("scene_source").as_string();
  if (source != "known" && source != "perception")
    throw std::runtime_error("[Scene] scene_source must be known or perception, got '" + source + "'");
  if (!std::isfinite(get_parameter("service_timeout").as_double()) ||
      get_parameter("service_timeout").as_double() <= 0.0)
    throw std::runtime_error("[Scene] service_timeout must be positive and finite");
  for (const auto& name : {"object_pose", "grasp_frame_transform", "place_pose"}) {
    const auto values = get_parameter(name).as_double_array();
    if ((values.size() != 6 && values.size() != 7) ||
        !std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); }))
      throw std::runtime_error(std::string("[Scene] ") + name + " requires xyz+rpy (6) or xyz+xyzw (7) finite values");
    if (values.size() == 7 && std::abs(Eigen::Quaterniond(values[6], values[3], values[4], values[5]).norm() - 1.0) > 1e-3)
      throw std::runtime_error(std::string("[Scene] ") + name + " quaternion must have unit norm");
  }
  if (get_parameter("drop_into_bin").as_bool()) {
    check_vector("bin_pose", 6);
    check_vector("bin_dimensions", 3, true);
    const auto dims = get_parameter("bin_dimensions").as_double_array();
    const double thickness = get_parameter("bin_wall_thickness").as_double();
    if (!std::isfinite(thickness) || thickness <= 0 ||
        dims[0] <= 2 * thickness || dims[1] <= 2 * thickness || dims[2] <= thickness)
      throw std::runtime_error("[Bin] Invalid wall thickness or bin dimensions");
    const Eigen::Vector3d release = vectorToEigen(get_parameter("bin_pose").as_double_array()).inverse() *
                                   vectorToEigen(get_parameter("place_pose").as_double_array()).translation();
    if (std::abs(release.x()) >= dims[0] / 2 - thickness ||
        std::abs(release.y()) >= dims[1] / 2 - thickness || release.z() <= dims[2])
      throw std::runtime_error("[Bin] place_pose must be above the open bin, with object bottom above the rim");
  }
  if (source == "known") {
    check_vector("support_surface_dimensions", 3, true);
    const auto pose = get_parameter("support_surface_pose").as_double_array();
    if ((pose.size() != 6 && pose.size() != 7) ||
        !std::all_of(pose.begin(), pose.end(), [](double v) { return std::isfinite(v); }) ||
        (pose.size() == 7 && std::abs(Eigen::Quaterniond(pose[6], pose[3], pose[4], pose[5]).norm() - 1.0) > 1e-3))
      throw std::runtime_error("[Scene] support_surface_pose requires a finite xyz+rpy or xyz+unit quaternion pose");
    const auto id = get_parameter("support_surface_name").as_string();
    if (id.empty() || id == get_parameter("object_name").as_string())
      throw std::runtime_error("[Scene] support_surface_name must be nonempty and different from object_name");
  }
  if (get_parameter("gripper_strategy").as_string() != "named_pose")
    throw std::runtime_error("[Gripper adapter] Only named_pose is implemented; supply a controller/action adapter for other strategies");
  if (get_parameter("object_name").as_string().empty())
    throw std::runtime_error("[Scene] object_name must not be empty");
  if (get_parameter("max_solutions").as_int() <= 0)
    throw std::runtime_error("[MTC configuration] max_solutions must be positive");

  task_.loadRobotModel(shared_from_this(), "robot_description");
  const auto model = task_.getRobotModel();
  if (!model) throw std::runtime_error("[Robot Description] Failed to load robot model");
  auto check_group = [&](const std::string& parameter) {
    const auto name = get_parameter(parameter).as_string();
    const auto* group = model->getJointModelGroup(name);
    if (!group) {
      std::string available;
      for (const auto& value : model->getJointModelGroupNames()) available += " " + value;
      throw std::runtime_error("[SRDF] " + parameter + "='" + name +
                               "' does not exist. Available groups:" + available);
    }
    return group;
  };
  const auto* arm = check_group("arm_group_name");
  const auto* gripper = check_group("gripper_group_name");
  auto check_pose = [&](const auto* group, const std::string& parameter) {
    const auto name = get_parameter(parameter).as_string();
    const auto& names = group->getDefaultStateNames();
    if (std::find(names.begin(), names.end(), name) == names.end()) {
      std::string available;
      for (const auto& value : names) available += " " + value;
      throw std::runtime_error("[SRDF] " + parameter + "='" + name +
          "' does not exist in group '" + group->getName() + "'. Available poses:" + available);
    }
  };
  check_pose(arm, "arm_home_pose");
  check_pose(gripper, "gripper_open_pose");
  check_pose(gripper, "gripper_close_pose");
  const auto eef_name = get_parameter("end_effector").as_string();
  const auto* eef = model->getEndEffector(eef_name);
  if (!eef || eef != gripper)
    throw std::runtime_error("[SRDF] end_effector='" + eef_name + "' does not refer to gripper_group_name");
  if (!arm->getSolverInstance())
    throw std::runtime_error("[IK] No kinematics solver available for arm_group_name");
  const auto pipeline = get_parameter("planning_pipeline").as_string();
  const auto planner = get_parameter("planner_id").as_string();
  if (!has_parameter(pipeline + ".planning_plugins") || !has_parameter(pipeline + ".planner_configs." + planner + ".type"))
    throw std::runtime_error("[Planner] Missing pipeline or planner configuration: " + pipeline + "/" + planner);

  const auto frame = get_parameter("gripper_frame").as_string();
  if (!model->hasLinkModel(frame))
    throw std::runtime_error("[Robot Description] gripper_frame='" + frame +
                             "' is not a robot link (required for IK and Attach/Detach)");
  for (const auto& parameter : {"world_frame", "object_reference_frame"})
    validateFrame(get_parameter(parameter).as_string(), parameter);
  if (get_parameter("scene_source").as_string() == "known")
    validateFrame(get_parameter("support_surface_reference_frame").as_string(), "support_surface_reference_frame");
  const auto controllers = get_parameter("controller_names").as_string_array();
  if (controllers.empty()) throw std::runtime_error("[Controller] controller_names is empty");
  for (const auto& name : controllers) {
    const auto parameter = "moveit_simple_controller_manager." + name + ".type";
    if (name.empty() || !has_parameter(parameter))
      throw std::runtime_error("[Controller] controller_names contains '" + name +
                               "', absent from the loaded MoveIt controller configuration");
    const auto joints_parameter = "moveit_simple_controller_manager." + name + ".joints";
    if (!has_parameter(joints_parameter))
      throw std::runtime_error("[Controller] Missing joints list for '" + name + "'");
    const auto joints = get_parameter(joints_parameter).as_string_array();
    for (const auto& joint : joints)
      if (!model->getJointModel(joint))
        throw std::runtime_error("[Controller] Unknown robot joint '" + joint + "' in '" + name + "'");
    const auto type = get_parameter(parameter).as_string();
    if (type != "FollowJointTrajectory" && type != "GripperCommand")
      throw std::runtime_error("[Gripper adapter] Unsupported MoveIt controller type '" + type + "'; an adapter is required");
  }
  RCLCPP_INFO(get_logger(), "[MTC configuration] Robot groups, frames, named poses and controllers validated");
}

void MTCTaskNode::validateFrame(const std::string& frame, const std::string& parameter)
{
  const auto model = task_.getRobotModel();
  if (!model) throw std::runtime_error("[Robot Description] No robot model for frame validation");
  if (!frame.empty() && (model->hasLinkModel(frame) || frame == model->getModelFrame())) return;
  tf2_ros::Buffer buffer(get_clock());
  tf2_ros::TransformListener listener(buffer);
  if (frame.empty() || !buffer.canTransform(model->getModelFrame(), frame, tf2::TimePointZero,
      tf2::durationFromSec(get_parameter("service_timeout").as_double())))
    throw std::runtime_error("[TF] " + parameter + "='" + frame + "' cannot transform to '" + model->getModelFrame() + "'");
}

/**
 * @brief Update the target object parameters after the service call to get planning scene
 */
void MTCTaskNode::updateObjectParameters(const moveit_msgs::msg::CollisionObject& collision_object) {
  if (collision_object.primitives.size() != 1 || collision_object.primitive_poses.size() != 1)
    throw std::runtime_error("[Scene] Target must contain one primitive and one pose");
  const auto& primitive = collision_object.primitives.front();
  const size_t expected = primitive.type == shape_msgs::msg::SolidPrimitive::CYLINDER ? 2 :
      primitive.type == shape_msgs::msg::SolidPrimitive::BOX ? 3 : 0;
  if (!expected || primitive.dimensions.size() != expected ||
      !std::all_of(primitive.dimensions.begin(), primitive.dimensions.end(),
                   [](double v) { return std::isfinite(v) && v > 0.0; }))
    throw std::runtime_error("[Scene] Invalid target type or dimensions");
  // Update object_name
  this->set_parameter(rclcpp::Parameter("object_name", collision_object.id));
  RCLCPP_INFO(this->get_logger(), "Updated object_name: '%s'", collision_object.id.c_str());

  // Update object_type and object_dimensions
  if (!collision_object.primitives.empty()) {
    std::vector<double> dimensions;
    std::string object_type;
    if (collision_object.primitives[0].type == shape_msgs::msg::SolidPrimitive::CYLINDER) {
      object_type = "cylinder";
      dimensions = {
        collision_object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT],
        collision_object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS]
      };
    } else if (collision_object.primitives[0].type == shape_msgs::msg::SolidPrimitive::BOX) {
      object_type = "box";
      dimensions = {
        collision_object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::BOX_X],
        collision_object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y],
        collision_object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z]
      };
    } else {
      RCLCPP_WARN(this->get_logger(), "Unsupported object type. Only cylinders and boxes are supported.");
      return;
    }
    this->set_parameter(rclcpp::Parameter("object_type", object_type));
    this->set_parameter(rclcpp::Parameter("object_dimensions", dimensions));
    RCLCPP_INFO(this->get_logger(), "Updated object_type: '%s'", object_type.c_str());
    RCLCPP_INFO(this->get_logger(), "Updated object_dimensions: [%s]",
                std::accumulate(std::next(dimensions.begin()), dimensions.end(),
                                std::to_string(dimensions[0]),
                                [](std::string a, double b) {
                                  return std::move(a) + ", " + std::to_string(b);
                                }).c_str());
  }
}

/**
 * @brief Set up the planning scene with collision objects.
 */
void MTCTaskNode::setupSceneFromKnownObjects()
{
  using Primitive = shape_msgs::msg::SolidPrimitive;
  auto make_object = [&](const std::string& id, const std::string& frame, uint8_t type,
                         const std::vector<double>& dimensions, const std::vector<double>& pose) {
    moveit_msgs::msg::CollisionObject object;
    object.id = id;
    object.header.frame_id = frame;
    object.operation = moveit_msgs::msg::CollisionObject::ADD;
    Primitive primitive;
    primitive.type = type;
    primitive.dimensions.assign(dimensions.begin(), dimensions.end());
    object.primitives.push_back(primitive);
    object.primitive_poses.push_back(vectorToPose(pose));
    return object;
  };
  target_object_id_ = get_parameter("object_name").as_string();
  support_surface_id_ = get_parameter("support_surface_name").as_string();
  scene_world_.collision_objects = {
    make_object(target_object_id_, get_parameter("object_reference_frame").as_string(),
                get_parameter("object_type").as_string() == "cylinder" ? Primitive::CYLINDER : Primitive::BOX,
                get_parameter("object_dimensions").as_double_array(), get_parameter("object_pose").as_double_array()),
    make_object(support_surface_id_, get_parameter("support_surface_reference_frame").as_string(), Primitive::BOX,
                get_parameter("support_surface_dimensions").as_double_array(), get_parameter("support_surface_pose").as_double_array())
  };
  const auto obstacles_file = get_parameter("known_obstacles_file").as_string();
  if (!obstacles_file.empty()) {
    std::ifstream file(obstacles_file, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() > 16 * 1024 * 1024)
      throw std::runtime_error("[Scene] Invalid known_obstacles_file: " + obstacles_file);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    rclcpp::SerializedMessage serialized(size);
    auto& bytes = serialized.get_rcl_serialized_message();
    if (!file.read(reinterpret_cast<char*>(bytes.buffer), size))
      throw std::runtime_error("[Scene] Could not read known_obstacles_file");
    bytes.buffer_length = size;
    moveit_msgs::msg::PlanningSceneWorld obstacles;
    rclcpp::Serialization<moveit_msgs::msg::PlanningSceneWorld> serializer;
    serializer.deserialize_message(&serialized, &obstacles);
    scene_world_.collision_objects.insert(scene_world_.collision_objects.end(),
                                         obstacles.collision_objects.begin(), obstacles.collision_objects.end());
  }
  RCLCPP_INFO(get_logger(), "[Scene] known: built target '%s' and support '%s'; no perception client created",
              target_object_id_.c_str(), support_surface_id_.c_str());
}

void MTCTaskNode::setupSceneFromPerception(const std::string& target_label)
{
  if (!planning_scene_client)
    planning_scene_client = std::make_shared<GetPlanningSceneClient>(
        get_parameter("scene_service_name").as_string(), get_parameter("service_timeout").as_double(),
        get_parameter("use_sim_time").as_bool());
  const auto response = planning_scene_client->call_service(
      get_parameter("object_type").as_string(), get_parameter("object_dimensions").as_double_array(), target_label);
  if (!response.success)
    throw std::runtime_error("[Scene] Perception returned success=false; planning stopped. Check perception server logs.");
  scene_world_ = response.scene_world;
  target_object_id_ = response.target_object_id;
  support_surface_id_ = response.support_surface_id;
  if (target_object_id_.empty() || support_surface_id_.empty() || target_object_id_ == support_surface_id_)
    throw std::runtime_error("[Scene] Perception returned empty or identical target/support IDs");
  const auto find = [&](const std::string& id) {
    return std::find_if(scene_world_.collision_objects.begin(), scene_world_.collision_objects.end(),
                        [&](const auto& object) { return object.id == id; });
  };
  const auto target = find(target_object_id_);
  if (target == scene_world_.collision_objects.end() || find(support_surface_id_) == scene_world_.collision_objects.end())
    throw std::runtime_error("[Scene] Missing target or support object in perception response");
  updateObjectParameters(*target);
}

void MTCTaskNode::setupPlanningScene(const std::string& target_label)
{
  scene_ready_ = false;
  if (get_parameter("scene_source").as_string() == "known") setupSceneFromKnownObjects();
  else setupSceneFromPerception(target_label);
  if (get_parameter("drop_into_bin").as_bool()) appendCollectionBin();
  std::set<std::string> scene_ids;
  for (const auto& object : scene_world_.collision_objects) {
    if (!scene_ids.insert(object.id).second)
      throw std::runtime_error("[Scene] Duplicate collision object ID: " + object.id);
    validateFrame(object.header.frame_id, "collision_object[" + object.id + "].frame");
    if (object.id.empty() || object.operation != moveit_msgs::msg::CollisionObject::ADD ||
        object.primitives.empty() || object.primitives.size() != object.primitive_poses.size())
      throw std::runtime_error("[Scene] Invalid collision object '" + object.id + "'");
    for (const auto& primitive : object.primitives) {
      const size_t count = primitive.type == shape_msgs::msg::SolidPrimitive::BOX ? 3 :
          primitive.type == shape_msgs::msg::SolidPrimitive::CYLINDER ? 2 :
          primitive.type == shape_msgs::msg::SolidPrimitive::SPHERE ? 1 : 0;
      if (!count || primitive.dimensions.size() != count ||
          !std::all_of(primitive.dimensions.begin(), primitive.dimensions.end(),
                       [](double value) { return std::isfinite(value) && value > 0.0; }))
        throw std::runtime_error("[Scene] Invalid primitive dimensions in '" + object.id + "'");
    }
    for (const auto& pose : object.primitive_poses) {
      const Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
      if (!std::isfinite(pose.position.x) || !std::isfinite(pose.position.y) || !std::isfinite(pose.position.z) ||
          !std::isfinite(q.norm()) || std::abs(q.norm() - 1.0) > 1e-3)
        throw std::runtime_error("[Scene] Invalid pose/quaternion in '" + object.id + "'");
    }
  }
  const auto timeout = std::chrono::duration<double>(get_parameter("service_timeout").as_double());
  auto apply = create_client<moveit_msgs::srv::ApplyPlanningScene>("apply_planning_scene");
  auto get = create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
  if (!apply->wait_for_service(timeout) || !get->wait_for_service(timeout))
    throw std::runtime_error("[MoveIt] Planning scene services unavailable before service_timeout");
  moveit::planning_interface::PlanningSceneInterface psi("", false);
  // Never discard an object still held after a failed execution. It needs recovery
  // before another capture/plan can safely replace its collision geometry.
  if (!psi.getAttachedObjects().empty())
    throw std::runtime_error("[Scene] An object is still attached; recover/release it before another task");
  auto objects = scene_world_.collision_objects;
  std::vector<moveit_msgs::msg::CollisionObject> refreshed;
  for (const auto& id : managed_scene_ids_) {
    moveit_msgs::msg::CollisionObject remove;
    remove.id = id;
    remove.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    refreshed.push_back(std::move(remove));
  }
  refreshed.insert(refreshed.end(), objects.begin(), objects.end());
  if (!psi.applyCollisionObjects(refreshed))
    throw std::runtime_error("[Scene] MoveIt rejected collision objects; planning stopped");
  managed_scene_ids_.clear();
  for (const auto& object : objects) managed_scene_ids_.insert(object.id);
  scene_ready_ = true;
  RCLCPP_INFO(get_logger(), "[Scene] Applied %zu collision objects; target='%s', support='%s'",
              scene_world_.collision_objects.size(), target_object_id_.c_str(), support_surface_id_.c_str());
}

void MTCTaskNode::appendCollectionBin()
{
  const auto size = get_parameter("bin_dimensions").as_double_array();
  const double t = get_parameter("bin_wall_thickness").as_double();
  const auto base = vectorToEigen(get_parameter("bin_pose").as_double_array());
  moveit_msgs::msg::CollisionObject bin;
  bin.id = "collection_bin";
  bin.header.frame_id = get_parameter("world_frame").as_string();
  bin.operation = moveit_msgs::msg::CollisionObject::ADD;
  const auto add = [&](double x, double y, double z, double sx, double sy, double sz) {
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {sx, sy, sz};
    bin.primitives.push_back(primitive);
    bin.primitive_poses.push_back(tf2::toMsg(Eigen::Isometry3d(base * Eigen::Translation3d(x, y, z))));
  };
  add(0, 0, t / 2, size[0], size[1], t);
  add(-(size[0] - t) / 2, 0, size[2] / 2, t, size[1], size[2]);
  add((size[0] - t) / 2, 0, size[2] / 2, t, size[1], size[2]);
  add(0, -(size[1] - t) / 2, size[2] / 2, size[0] - 2 * t, t, size[2]);
  add(0, (size[1] - t) / 2, size[2] / 2, size[0] - 2 * t, t, size[2]);
  scene_world_.collision_objects.push_back(std::move(bin));
}

void MTCTaskNode::startCommandService()
{
  // Keep service execution separate from the default group used by MoveIt clients.
  // A second command can enter only to receive an immediate "busy" response.
  command_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  command_service_ = create_service<robot_task_interfaces::srv::RunPickPlace>(
      get_parameter("command_service_name").as_string(),
      [this](const robot_task_interfaces::srv::RunPickPlace::Request::SharedPtr request,
             robot_task_interfaces::srv::RunPickPlace::Response::SharedPtr response) {
        std::unique_lock<std::mutex> lock(command_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
          response->message = "Robot is busy; wait for the current task to finish";
          return;
        }
        if (request->target_label.find_first_not_of(" \t\r\n") == std::string::npos) {
          response->message = "Target text must not be empty";
          return;
        }
        RCLCPP_INFO(get_logger(), "[Command] Starting target='%s'", request->target_label.c_str());
        try {
          setupPlanningScene(request->target_label);
          doTask();
          response->success = true;
          response->executed = get_parameter("execute").as_bool();
          response->message = response->executed ? "Pick and place executed successfully" :
                                                  "Pick and place planned successfully (execute=false)";
        } catch (const std::exception& error) {
          scene_ready_ = false;
          response->message = error.what();
          RCLCPP_ERROR(get_logger(), "[Command] Failed: %s", error.what());
        }
        RCLCPP_INFO(get_logger(), "[Command] Finished success=%s; waiting for next command",
                    response->success ? "true" : "false");
      }, rclcpp::ServicesQoS(), command_group_);
  RCLCPP_INFO(get_logger(), "[Command] Ready on %s; no perception or planning until a command arrives",
              get_parameter("command_service_name").as_string().c_str());
}

void MTCTaskNode::checkExecutionControllers()
{
  const auto timeout = std::chrono::duration<double>(get_parameter("service_timeout").as_double());
  auto client = create_client<controller_manager_msgs::srv::ListControllers>(
      get_parameter("controller_manager").as_string() + "/list_controllers");
  if (!client->wait_for_service(timeout)) throw std::runtime_error("[Controller] Controller manager unavailable");
  auto future = client->async_send_request(std::make_shared<controller_manager_msgs::srv::ListControllers::Request>());
  if (future.wait_for(timeout) != std::future_status::ready)
    throw std::runtime_error("[Controller] list_controllers response timed out");
  const auto response = future.get();
  const auto controllers = get_parameter("controller_names").as_string_array();
  for (const auto& name : controllers) {
    const auto active = std::find_if(response->controller.begin(), response->controller.end(),
        [&](const auto& controller) { return controller.name == name && controller.state == "active"; });
    if (active == response->controller.end())
      throw std::runtime_error("[Controller] '" + name + "' is not active; execution stopped");
    const auto prefix = "moveit_simple_controller_manager." + name;
    const auto action = name + "/" + get_parameter(prefix + ".action_ns").as_string();
    const auto type = get_parameter(prefix + ".type").as_string();
    bool ready = false;
    if (type == "FollowJointTrajectory") {
      auto action_client = rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(shared_from_this(), action);
      ready = action_client->wait_for_action_server(timeout);
    } else if (type == "GripperCommand") {
      auto action_client = rclcpp_action::create_client<control_msgs::action::GripperCommand>(shared_from_this(), action);
      ready = action_client->wait_for_action_server(timeout);
    }
    if (!ready) throw std::runtime_error("[Controller] Action unavailable: " + action);
  }
}

/**
 * @brief Plan and/or execute the pick and place task.
 */
void MTCTaskNode::saveDemonstration(const std::string& state, int code)
{
  const auto directory = get_parameter("demonstration_directory").as_string();
  if (directory.empty()) return;
  const std::filesystem::path root(directory);
  std::filesystem::create_directories(root);
  // Preserve the exact selected solution, including scene diffs and gripper
  // trajectories. CDR can be deserialized by the matching ROS message version.
  if (state == "planned") {
    auto write_message = [&root](const std::string& name, const auto& message) {
      using Message = std::decay_t<decltype(message)>;
      rclcpp::Serialization<Message> serializer;
      rclcpp::SerializedMessage serialized;
      serializer.serialize_message(&message, &serialized);
      const auto& bytes = serialized.get_rcl_serialized_message();
      std::ofstream file(root / name, std::ios::binary);
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.write(reinterpret_cast<const char*>(bytes.buffer), bytes.buffer_length);
    };
    moveit_task_constructor_msgs::msg::Solution solution;
    task_.solutions().front()->toMsg(solution, &task_.introspection());
    write_message("solution.cdr", solution);
    moveit_task_constructor_msgs::msg::TaskDescription description;
    task_.introspection().fillTaskDescription(description);
    write_message("task_description.cdr", description);
  }
  // Rename only after close, so the collector never sees a partial status file.
  {
    std::ofstream file(root / "task_status.json.tmp");
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << "{\"state\":\"" << state << "\",\"moveit_error_code\":" << code
         << ",\"execution_start_ns\":" << execution_start_ns_
         << ",\"execution_end_ns\":" << execution_end_ns_ << "}\n";
  }
  std::filesystem::rename(root / "task_status.json.tmp", root / "task_status.json");
}

void MTCTaskNode::doTask()
{
  if (!scene_ready_) throw std::runtime_error("[Scene] Cannot plan without a valid scene");
  RCLCPP_INFO(this->get_logger(), "Starting the pick and place task");

  task_ = createTask();

  // Get parameters
  auto execute = this->get_parameter("execute").as_bool();
  auto max_solutions = this->get_parameter("max_solutions").as_int();

  try
  {
    task_.init();
    RCLCPP_INFO(this->get_logger(), "Task initialized successfully");
  }
  catch (mtc::InitStageException& e)
  {
    throw std::runtime_error(std::string("[MTC initialization] ") + e.what());
  }

  // Attempt to plan the task
  if (!task_.plan(max_solutions))
  {
    task_.explainFailure();
    task_.stages()->traverseRecursively([this](const mtc::Stage& stage, unsigned int) {
      if (stage.name() != "close gripper" || stage.failures().empty()) return true;
      const auto& failure = stage.failures().front();
      if (!failure->start()) return true;
      const auto scene = failure->start()->scene();
      auto state = scene->getCurrentState();
      state.setToDefaultValues(get_parameter("gripper_group_name").as_string(), get_parameter("gripper_close_pose").as_string());
      state.update();
      collision_detection::CollisionRequest request;
      request.contacts = true;
      request.max_contacts = 10;
      collision_detection::CollisionResult result;
      scene->checkCollision(request, result, state);
      for (const auto& contact : result.contacts)
        RCLCPP_ERROR(get_logger(), "[MTC collision] close gripper: %s <-> %s", contact.first.first.c_str(), contact.first.second.c_str());
      return true;
    });
    throw std::runtime_error("[MTC planning] Task planning failed");
  }

  RCLCPP_INFO(this->get_logger(), "Task planning succeeded");

  // Publish the planned solution for visualization
  task_.introspection().publishSolution(*task_.solutions().front());
  RCLCPP_INFO(this->get_logger(), "Published solution for visualization");
  saveDemonstration("planned");

  if (execute)
  {
    checkExecutionControllers();
    // Execute the planned task
    RCLCPP_INFO(this->get_logger(), "Executing the planned task");
    execution_start_ns_ = now().nanoseconds();
    saveDemonstration("executing");
    auto result = task_.execute(*task_.solutions().front());
    execution_end_ns_ = now().nanoseconds();
    saveDemonstration(result.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS ?
                      "executed" : "execution_failed", result.val);
    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      throw std::runtime_error("[MTC execution] Failed with code " + std::to_string(result.val));
    }
    RCLCPP_INFO(this->get_logger(), "Task executed successfully");
    if (get_parameter("drop_into_bin").as_bool() && !get_parameter("pick_only").as_bool()) {
      // The released object falls under gravity; do not retain a floating estimate.
      moveit_msgs::msg::CollisionObject remove;
      remove.id = target_object_id_;
      remove.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      moveit::planning_interface::PlanningSceneInterface psi("", false);
      if (psi.applyCollisionObjects({remove})) managed_scene_ids_.erase(target_object_id_);
      else RCLCPP_WARN(get_logger(), "[Bin] Could not clear the released object's estimated pose");
    }
  }
  else
  {
    RCLCPP_INFO(this->get_logger(), "Execution skipped as per configuration");
  }

  return;
}

/**
 * @brief Create the MTC task with all necessary stages.
 * @return The created MTC task.
 */
mtc::Task MTCTaskNode::createTask()
{
  RCLCPP_INFO(this->get_logger(), "Creating MTC task");

  // Create a new Task
  mtc::Task task;

  // Set the name of the task
  task.stages()->setName("pick_place_task");

  // Load the robot model into the task
  task.loadRobotModel(shared_from_this(), "robot_description");

  // Get parameters
  // Robot configuration parameters
  auto arm_group_name = this->get_parameter("arm_group_name").as_string();
  auto gripper_group_name = this->get_parameter("gripper_group_name").as_string();
  const auto touch_links = gripperTouchLinks(task.getRobotModel(), gripper_group_name);
  auto gripper_frame = this->get_parameter("gripper_frame").as_string();
  auto arm_home_pose = this->get_parameter("arm_home_pose").as_string();

  // Gripper poses
  auto gripper_open_pose = this->get_parameter("gripper_open_pose").as_string();
  auto gripper_close_pose = this->get_parameter("gripper_close_pose").as_string();

  // Frame parameters
  auto world_frame = this->get_parameter("world_frame").as_string();

  // Controller parameters
  auto controller_names = this->get_parameter("controller_names").as_string_array();

  // Object parameters
  auto object_name = this->get_parameter("object_name").as_string();
  auto object_type = this->get_parameter("object_type").as_string();
  auto object_reference_frame = this->get_parameter("object_reference_frame").as_string();
  auto object_dimensions = this->get_parameter("object_dimensions").as_double_array();
  auto object_pose = this->get_parameter("object_pose").as_double_array();

  RCLCPP_INFO(this->get_logger(), "Creating task for object:");
  RCLCPP_INFO(this->get_logger(), "  Name: %s", object_name.c_str());
  RCLCPP_INFO(this->get_logger(), "  Type: %s", object_type.c_str());
  RCLCPP_INFO(this->get_logger(), "  Dimensions: [%s]",
              std::accumulate(std::next(object_dimensions.begin()), object_dimensions.end(),
                              std::to_string(object_dimensions[0]),
                              [](std::string a, double b) {
                                return std::move(a) + ", " + std::to_string(b);
                              }).c_str());

  // Grasp and place parameters
  auto grasp_frame_transform = this->get_parameter("grasp_frame_transform").as_double_array();
  auto place_pose = this->get_parameter("place_pose").as_double_array();

  // Motion planning parameters
  auto approach_object_min_dist = this->get_parameter("approach_object_min_dist").as_double();
  auto approach_object_max_dist = this->get_parameter("approach_object_max_dist").as_double();
  auto lift_object_min_dist = this->get_parameter("lift_object_min_dist").as_double();
  auto lift_object_max_dist = this->get_parameter("lift_object_max_dist").as_double();
  auto lower_object_min_dist = this->get_parameter("lower_object_min_dist").as_double();
  auto lower_object_max_dist = this->get_parameter("lower_object_max_dist").as_double();

  // Timeout parameters
  auto move_to_pick_timeout = this->get_parameter("move_to_pick_timeout").as_double();
  auto move_to_place_timeout = this->get_parameter("move_to_place_timeout").as_double();

  // Grasp generation parameters
  auto grasp_pose_angle_delta = this->get_parameter("grasp_pose_angle_delta").as_double();
  auto grasp_pose_max_ik_solutions = this->get_parameter("grasp_pose_max_ik_solutions").as_int();
  auto grasp_pose_min_solution_distance = this->get_parameter("grasp_pose_min_solution_distance").as_double();

  // Place generation parameters
  auto place_pose_max_ik_solutions = this->get_parameter("place_pose_max_ik_solutions").as_int();

  // Cartesian planner parameters
  auto cartesian_max_velocity_scaling = this->get_parameter("cartesian_max_velocity_scaling").as_double();
  auto cartesian_max_acceleration_scaling = this->get_parameter("cartesian_max_acceleration_scaling").as_double();
  auto cartesian_step_size = this->get_parameter("cartesian_step_size").as_double();

  // Direction vector parameters
  auto approach_object_direction_z = this->get_parameter("approach_object_direction_z").as_double();
  auto lift_object_direction_z = this->get_parameter("lift_object_direction_z").as_double();
  auto lower_object_direction_z = this->get_parameter("lower_object_direction_z").as_double();
  auto retreat_direction_z = this->get_parameter("retreat_direction_z").as_double();

  // Other parameters
  auto place_pose_z_offset_factor = this->get_parameter("place_pose_z_offset_factor").as_double();
  auto retreat_min_distance = this->get_parameter("retreat_min_distance").as_double();
  auto retreat_max_distance = this->get_parameter("retreat_max_distance").as_double();

  // Create planners for different types of motion
  // Pipeline planner for complex movements
  // OMPL planner
  std::unordered_map<std::string, std::string> ompl_map_arm = {
    {get_parameter("planning_pipeline").as_string(),
     arm_group_name + "[" + get_parameter("planner_id").as_string() + "]"}
  };
  auto ompl_planner_arm = std::make_shared<mtc::solvers::PipelinePlanner>(
    this->shared_from_this(),
    ompl_map_arm);
  RCLCPP_INFO(this->get_logger(), "OMPL planner created for the arm group");

  // JointInterpolation is a basic planner that is used for simple motions
  // It computes quickly but doesn't support complex motions.
  auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
  RCLCPP_INFO(this->get_logger(), "Joint Interpolation planner created for the gripper group");

  // Cartesian planner
  auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
  cartesian_planner->setMaxVelocityScalingFactor(cartesian_max_velocity_scaling);
  cartesian_planner->setMaxAccelerationScalingFactor(cartesian_max_acceleration_scaling);
  cartesian_planner->setStepSize(cartesian_step_size);
  RCLCPP_INFO(this->get_logger(), "Cartesian planner created");

  // Set task properties
  task.setProperty("trajectory_execution_info",
    mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
  task.setProperty("group", arm_group_name); // The main planning group
  task.setProperty("eef", get_parameter("end_effector").as_string()); // The end-effector group
  task.setProperty("ik_frame", gripper_frame); // The frame for inverse kinematics

  /****************************************************
   *                                                  *
   *               Current State                      *
   *                                                  *
   ***************************************************/
  // Pointer to store the current state (will be used during the grasp pose generation stage)
  mtc::Stage* current_state_ptr = nullptr;

  // Add a stage to capture the current state
  auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current state");
  current_state_ptr = stage_state_current.get();
  task.add(std::move(stage_state_current));

  /****************************************************
   *                                                  *
   *               Open Gripper                       *
   *                                                  *
   ***************************************************/
  // This stage is responsible for opening the robot's gripper in preparation for picking
  // up an object in the pick-and-place task.
  auto stage_open_gripper =
    std::make_unique<mtc::stages::MoveTo>("open gripper", interpolation_planner);
  stage_open_gripper->setGroup(gripper_group_name);
  stage_open_gripper->setGoal(gripper_open_pose);
  stage_open_gripper->properties().set("trajectory_execution_info",
                      mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
  task.add(std::move(stage_open_gripper));

  /****************************************************
   *                                                  *
   *               Move to Pick                       *
   *                                                  *
   ***************************************************/
  // Create a stage to move the arm to a pre-grasp position
  auto stage_move_to_pick = std::make_unique<mtc::stages::Connect>(
      "move to pick",
      mtc::stages::Connect::GroupPlannerVector{
        {arm_group_name, ompl_planner_arm},
        {gripper_group_name, interpolation_planner}
      });
  stage_move_to_pick->setTimeout(move_to_pick_timeout);
  stage_move_to_pick->properties().configureInitFrom(mtc::Stage::PARENT);
  task.add(std::move(stage_move_to_pick));

  // Create a pointer for the stage that will attach the object (to be used later)
  // By declaring it at the top level of the function, it can be accessed throughout
  // the entire task creation process.
  // This allows different parts of the code to use and modify this pointer.
  mtc::Stage* attach_object_stage =
      nullptr;  // Forward attach_object_stage to place pose generator

  /****************************************************
   *                                                  *
   *               Pick Object                        *
   *                                                  *
   ***************************************************/
  {
    // Create a serial container for the grasping action
    // This container will hold stages (in order) that will accomplish the picking action
    auto grasp = std::make_unique<mtc::SerialContainer>("pick object");
    task.properties().exposeTo(grasp->properties(), { "eef", "group", "ik_frame" });
    grasp->properties().configureInitFrom(mtc::Stage::PARENT,
                                        { "eef", "group", "ik_frame" });

    /****************************************************
---- *               Approach Object                    *
     ***************************************************/
    {
      // Create a stage for moving the gripper close to the object before trying to grab it.
      // We are doing a movement that is relative to our current position.
      // Cartesian planner will move the gripper in a straight line
      auto stage =
        std::make_unique<mtc::stages::MoveRelative>("approach object", cartesian_planner);

      // Set properties for visualization and planning
      stage->properties().set("marker_ns", "approach_object"); // Namespace for visualization markers
      stage->properties().set("link", gripper_frame); // The link to move (end effector)
      stage->properties().set("trajectory_execution_info",
                      mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" }); // Inherit the 'group' property
      stage->setMinMaxDistance(approach_object_min_dist, approach_object_max_dist);

      // Define the direction that we want the gripper to move (i.e. z direction) from the gripper frame
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = gripper_frame; // Set the frame for the vector
      vec.vector.z = approach_object_direction_z; // Set the direction (in this case, along the z-axis of the gripper frame)
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }

    /****************************************************
---- *               Generate Grasp Pose               *
     ***************************************************/
    {
	  // Generate the grasp pose
	  // This is the stage for computing how the robot should grab the object
	  // This stage is a generator stage because it doesn't need information from
	  // stages before or after it.
	  // When generating solutions, MTC will try to grab the object from many different orientations.
      // Sample grasp pose candidates in angle increments around the z-axis of the object

      auto stage = std::make_unique<mtc::stages::GenerateGraspPose>("generate grasp pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "grasp_pose");
      stage->setPreGraspPose(gripper_open_pose);
      stage->setObject(object_name);
      stage->setAngleDelta(grasp_pose_angle_delta); //  Angular resolution for sampling grasp poses around the object
      stage->setMonitoredStage(current_state_ptr);  // Ensure grasp poses are valid given the initial configuration of the robot

      // Compute IK for sampled grasp poses
      auto wrapper = std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(grasp_pose_max_ik_solutions);
      wrapper->setMinSolutionDistance(grasp_pose_min_solution_distance);
      wrapper->setIKFrame(vectorToEigen(grasp_frame_transform), gripper_frame); // Transform from gripper frame to tool center point (TCP)
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      grasp->insert(std::move(wrapper));
    }

    /****************************************************
---- *            Allow Collision (gripper,  object)   *
     ***************************************************/
    {
      // Modify planning scene (w/o altering the robot's pose) to allow touching the object for picking
      auto stage =
        std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (gripper,object)");
      // Include moving mimic/descendant fingers, which may be outside the SRDF group.
      stage->allowCollisions(
        object_name,
        touch_links,
        true);
      grasp->insert(std::move(stage));
    }

    /****************************************************
---- *               Close Gripper                     *
     ***************************************************/
    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("close gripper", interpolation_planner);
      stage->setGroup(gripper_group_name);
      stage->setGoal(gripper_close_pose);
      stage->properties().set("trajectory_execution_info",
                      mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
      grasp->insert(std::move(stage));
    }

    /****************************************************
---- *       Allow collision (object,  support_surface)        *
     ***************************************************/
    {
      // Allows the planner to generate valid trajectories where the object remains in contact
      // with the support surface until it's lifted.
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (object,support_surface)");
      stage->allowCollisions({ object_name }, {support_surface_id_}, true);
      grasp->insert(std::move(stage));
    }

    /****************************************************
---- *               Attach Object                     *
     ***************************************************/
    {
       auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach object");
       stage->attachObject(object_name, gripper_frame);  // attach object to gripper_frame
       attach_object_stage = stage.get();
       grasp->insert(std::move(stage));
    }

    /****************************************************
---- *       Lift object                               *
     ***************************************************/
    {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("lift object", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setMinMaxDistance(lift_object_min_dist, lift_object_max_dist);
      stage->setIKFrame(gripper_frame);
      stage->properties().set("marker_ns", "lift_object");
      stage->properties().set("trajectory_execution_info",
                      mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));

      // We're defining the direction to lift the object
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = world_frame;
      vec.vector.z = lift_object_direction_z;  // This means "straight up"
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }
    /****************************************************
---- *       Forbid collision (object, surface)*       *
     ***************************************************/
    {
      // Forbid collisions between the picked object and the support surface.
      // This is important after the object has been lifted to ensure it doesn't accidentally
      // collide with the surface during subsequent movements.
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (object,support_surface)");
      stage->allowCollisions({ object_name }, {support_surface_id_}, false);
      grasp->insert(std::move(stage));
    }

      // Add the serial container to the robot's to-do list
      // This serial container contains all the sequential steps we've created for grasping
      // and lifting the object
      task.add(std::move(grasp));
  }
  if (get_parameter("pick_only").as_bool()) return task;
  /******************************************************
   *                                                    *
   *          Move to Place                             *
   *                                                    *
   *****************************************************/
  {
    // Connect the grasped state to the pre-place state, i.e. realize the object transport
    // In other words, this stage plans the motion that transports the object from where it was picked up
    // to where it will be placed.
    auto stage_move_to_place = std::make_unique<mtc::stages::Connect>(
      "move to place",
      mtc::stages::Connect::GroupPlannerVector{
        {arm_group_name, ompl_planner_arm},
        {gripper_group_name, interpolation_planner}
      });
    stage_move_to_place->setTimeout(move_to_place_timeout);
    stage_move_to_place->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(stage_move_to_place));
  }

  /******************************************************
   *                                                    *
   *          Place Object                              *
   *                                                    *
   *****************************************************/
   // All placing sub-stages are collected within a serial container
  {
    auto place = std::make_unique<mtc::SerialContainer>("place object");
    task.properties().exposeTo(place->properties(), { "eef", "group", "ik_frame" });
    place->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group", "ik_frame" });

    /******************************************************
---- *          Lower Object                              *
     *****************************************************/
    if (!get_parameter("drop_into_bin").as_bool()) {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("lower object", cartesian_planner);
      stage->properties().set("marker_ns", "lower_object");
      stage->properties().set("link", gripper_frame);
      stage->properties().set("trajectory_execution_info",
                      mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setMinMaxDistance(lower_object_min_dist, lower_object_max_dist);

      // Set downward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = world_frame;
      vec.vector.z = lower_object_direction_z;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }

    /******************************************************
---- *          Generate Place Pose                       *
     *****************************************************/
    {
      // Generate Place Pose
      auto stage = std::make_unique<mtc::stages::GeneratePlacePose>("generate place pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "ik_frame" });
      stage->properties().set("marker_ns", "place_pose");
      stage->setObject(object_name);

      // Set target pose
      geometry_msgs::msg::PoseStamped target_pose_msg;
      target_pose_msg.header.frame_id = world_frame;
      target_pose_msg.pose = vectorToPose(place_pose);
      // Project the object's dimensions onto the place frame Z axis, including orientation.
      const Eigen::Matrix3d rotation = vectorToEigen(place_pose).rotation();
      const double height = object_type == "box" ?
          std::abs(rotation(2, 0)) * object_dimensions[0] + std::abs(rotation(2, 1)) * object_dimensions[1] +
          std::abs(rotation(2, 2)) * object_dimensions[2] :
          std::abs(rotation(2, 2)) * object_dimensions[0] +
          2.0 * object_dimensions[1] * std::hypot(rotation(2, 0), rotation(2, 1));
      target_pose_msg.pose.position.z += place_pose_z_offset_factor * height;
      RCLCPP_INFO(get_logger(), "[Place IK] Target object center in %s: [%.4f, %.4f, %.4f]",
                  world_frame.c_str(), target_pose_msg.pose.position.x, target_pose_msg.pose.position.y,
                  target_pose_msg.pose.position.z);
      stage->setPose(target_pose_msg);
      stage->setMonitoredStage(attach_object_stage);  // hook into successful pick solutions

      // Compute IK
      auto wrapper = std::make_unique<mtc::stages::ComputeIK>("place pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(place_pose_max_ik_solutions);
      wrapper->setIKFrame(object_name);  // GeneratePlacePose targets the attached object frame, including the actual sampled grasp.
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      place->insert(std::move(wrapper));
    }

    /******************************************************
---- *          Open Gripper                              *
     *****************************************************/
    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open gripper", interpolation_planner);
      stage->setGroup(gripper_group_name);
      stage->setGoal(gripper_open_pose);
      stage->properties().set("trajectory_execution_info",
        mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
      place->insert(std::move(stage));
    }

    /******************************************************
---- *          Forbid collision (gripper, object)        *
     *****************************************************/
    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (gripper,object)");
      stage->allowCollisions(object_name,
        touch_links, false);
      place->insert(std::move(stage));
    }

    /******************************************************
---- *          Detach Object                             *
     *****************************************************/
    {
      // Update the planning scene to reflect that the object is no longer attached to the gripper.
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("detach object");
      stage->detachObject(object_name, gripper_frame);
      place->insert(std::move(stage));
    }

    /******************************************************
---- *          Retreat Motion                            *
     *****************************************************/
    {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("retreat after place", cartesian_planner);
      stage->properties().set("trajectory_execution_info",
        mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setMinMaxDistance(retreat_min_distance, retreat_max_distance);
      stage->setIKFrame(gripper_frame);
      stage->properties().set("marker_ns", "retreat");
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = gripper_frame;
      vec.vector.z = retreat_direction_z;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }

    // Add place container to task
    task.add(std::move(place));
  }

  /******************************************************
   *                                                    *
   *          Move to Home                              *
   *                                                    *
   *****************************************************/
  {
    auto stage = std::make_unique<mtc::stages::MoveTo>("move home", ompl_planner_arm);
    stage->properties().set("trajectory_execution_info",
                      mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
    stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
    stage->setGoal(arm_home_pose);
    task.add(std::move(stage));
  }
  return task;
}

/**
 * @brief Main function to run the MTC task node.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit status.
 */
int main(int argc, char** argv)
{
  // Initialize ROS 2
  rclcpp::init(argc, argv);

  int ret = 0;

  try {
    // Set up node options
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    // Create the MTC task node
    auto mtc_task_node = std::make_shared<MTCTaskNode>(options);

    // Set up a multi-threaded executor
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
    executor.add_node(mtc_task_node);

    std::thread spinner([&executor]() { executor.spin(); });
    while (!executor.is_spinning()) std::this_thread::yield();
    // Set up the planning scene and execute the task
    try {
      mtc_task_node->validateConfiguration();
      if (mtc_task_node->get_parameter("interactive").as_bool()) {
        mtc_task_node->startCommandService();
      } else {
        RCLCPP_INFO(mtc_task_node->get_logger(), "Setting up planning scene");
        mtc_task_node->setupPlanningScene();
        RCLCPP_INFO(mtc_task_node->get_logger(), "Executing task");
        mtc_task_node->doTask();
        RCLCPP_INFO(mtc_task_node->get_logger(), "Task execution completed. Keeping node alive for visualization. Press Ctrl+C to exit.");
      }

      // Keep the node running until Ctrl+C is pressed
      spinner.join();
    } catch (const std::runtime_error& e) {
      RCLCPP_ERROR(mtc_task_node->get_logger(), "Runtime error occurred: %s", e.what());
      ret = 1;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(mtc_task_node->get_logger(), "An error occurred: %s", e.what());
      ret = 1;
    }
    executor.cancel();
    if (spinner.joinable()) spinner.join();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("main"), "Error during node setup: %s", e.what());
    ret = 1;
  }

  // Cleanup
  rclcpp::shutdown();
  return ret;
}
