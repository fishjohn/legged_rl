//
// Created by luohx on 23-11-15.
//

#include "legged_rl_controllers/BipedController.h"

#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include <pluginlib/class_list_macros.hpp>

namespace legged {
bool BipedController::init(hardware_interface::RobotHW* robotHw, ros::NodeHandle& controllerNH) {
  // Load policy model and rl cfg
  if (!loadModel(controllerNH)) {
    ROS_ERROR_STREAM("[RLControllerBase] Failed to load the model. Ensure the path is correct and accessible.");
    return false;
  }
  if (!loadRLCfg(controllerNH)) {
    ROS_ERROR_STREAM("[RLControllerBase] Failed to load the rl config. Ensure the yaml is correct and accessible.");
    return false;
  }

  // Hardware interface
  auto* hybridJointInterface = robotHw->get<HybridJointInterface>();
  std::vector<std::string> jointNames = {"abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"};
  for (const auto& jointName : jointNames) {
    hybridJointHandles_.push_back(hybridJointInterface->getHandle(jointName));
  }
  imuSensorHandles_ = robotHw->get<hardware_interface::ImuSensorInterface>()->getHandle("unitree_imu");

  auto* contactInterface = robotHw->get<ContactSensorInterface>();
  std::vector<std::string> footNames = {"foot_L_Link", "foot_R_Link"};
  for (const auto& footName : footNames) {
    contactHandles_.push_back(contactInterface->getHandle(footName));
  }

  cmdVelSub_ = controllerNH.subscribe("/cmd_vel", 1, &BipedController::cmdVelCallback, this);
  gTStateSub_ = controllerNH.subscribe("/ground_truth/state", 1, &BipedController::stateUpdateCallback, this);

  return true;
}

void BipedController::starting(const ros::Time& time) {
  loopCount_ = 0;
}

void BipedController::update(const ros::Time& time, const ros::Duration& period) {
  // compute observation & actions
  if (loopCount_ % robotCfg_.controlCfg.decimation == 0) {
    computeObservation();
    computeActions();
    // limit action range
    scalar_t actionMin = -robotCfg_.clipActions;
    scalar_t actionMax = robotCfg_.clipActions;
    std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                   [actionMin, actionMax](scalar_t x) { return std::max(actionMin, std::min(actionMax, x)); });
  }

  // set action
  for (int i = 0; i < hybridJointHandles_.size(); i++) {
    scalar_t pos_des = actions_[i] * robotCfg_.controlCfg.actionScale + defaultJointAngles_(i, 0);
    hybridJointHandles_[i].setCommand(pos_des, 0, robotCfg_.controlCfg.stiffness, robotCfg_.controlCfg.damping, 0);
    lastActions_(i, 0) = actions_[i];
  }

  loopCount_++;
}

void BipedController::computeActions() {
  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observations_.data(), observations_.size(),
                                                                   inputShapes_[0].data(), inputShapes_[0].size()));
  // run inference
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = sessionPtr_->Run(runOptions, inputNames_.data(), inputValues.data(), 1, outputNames_.data(), 1);

  for (int i = 0; i < actionsSize_; i++) {
    actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void BipedController::computeObservation() {
  Eigen::Quaternion<scalar_t> quat;
  for (size_t i = 0; i < 4; ++i) {
    quat.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
  }

  vector3_t zyx = quatToZyx(quat);
  matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();

  // linear velocity (base frame)
  vector3_t baseLinVel = inverseRot * baseLinVel_;
  // Angular velocity
  vector3_t baseAngVel(imuSensorHandles_.getAngularVelocity()[0], imuSensorHandles_.getAngularVelocity()[1],
                       imuSensorHandles_.getAngularVelocity()[2]);

  // Projected gravity
  vector3_t gravityVector(0, 0, -1);
  vector3_t projectedGravity(inverseRot * gravityVector);

  // command
  vector3_t command = command_;

  // dof position and dof velocity
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  for (size_t i = 0; i < hybridJointHandles_.size(); ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }

  vector_t gait(4);
  gait << 2.0, 0.5, 0.5, 0.1;  // trot
  gait_index_ += 0.02 * gait(0);
  if (gait_index_ > 1.0) {
    gait_index_ = 0.0;
  }
  vector_t gait_clock(2);
  gait_clock << sin(gait_index_ * 2 * M_PI), cos(gait_index_ * 2 * M_PI);

  // actions
  vector_t actions(lastActions_);

  RLRobotCfg::ObsScales& obsScales = robotCfg_.obsScales;
  matrix_t commandScaler = Eigen::DiagonalMatrix<scalar_t, 3>(obsScales.linVel, obsScales.linVel, obsScales.angVel);

  vector_t obs(observationSize_);
  // clang-format off
  obs << projectedGravity,
      baseAngVel,
      (jointPos - defaultJointAngles_) * obsScales.dofPos,
      jointVel * obsScales.dofVel,
      commandScaler * command, 0.7,
      actions,
      gait_clock,
      gait;
  // clang-format on

  for (size_t i = 0; i < obs.size(); i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
  }
  // Limit observation range
  scalar_t obsMin = -robotCfg_.clipObs;
  scalar_t obsMax = robotCfg_.clipObs;
  std::transform(observations_.begin(), observations_.end(), observations_.begin(),
                 [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

bool BipedController::loadModel(ros::NodeHandle& nh) {
  std::string policyFilePath;
  if (!nh.getParam("/policyFile", policyFilePath)) {
    ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
    return false;
  }

  policyFilePath_ = policyFilePath;
  ROS_INFO_STREAM("Load Onnx model from path : " << policyFilePath);

  // create env
  onnxEnvPrt_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
  // create session
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetInterOpNumThreads(1);
  sessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyFilePath.c_str(), sessionOptions);
  // get input and output info
  inputNames_.clear();
  outputNames_.clear();
  inputShapes_.clear();
  outputShapes_.clear();
  Ort::AllocatorWithDefaultOptions allocator;
  for (int i = 0; i < sessionPtr_->GetInputCount(); i++) {
    inputNames_.push_back(sessionPtr_->GetInputName(i, allocator));
    inputShapes_.push_back(sessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (int i = 0; i < sessionPtr_->GetOutputCount(); i++) {
    outputNames_.push_back(sessionPtr_->GetOutputName(i, allocator));
    outputShapes_.push_back(sessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  ROS_INFO_STREAM("Load Onnx model from successfully !!!");
  return true;
}

bool BipedController::loadRLCfg(ros::NodeHandle& nh) {
  RLRobotCfg::InitState& initState = robotCfg_.initState;
  RLRobotCfg::ControlCfg& controlCfg = robotCfg_.controlCfg;
  RLRobotCfg::ObsScales& obsScales = robotCfg_.obsScales;

  int error = 0;
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/L_HAA_joint", initState.L_HAA_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/L_HFE_joint", initState.L_HFE_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/L_KFE_joint", initState.L_KFE_joint));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/R_HAA_joint", initState.R_HAA_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/R_HFE_joint", initState.R_HFE_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/R_KFE_joint", initState.R_KFE_joint));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/stiffness", controlCfg.stiffness));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/damping", controlCfg.damping));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/action_scale", controlCfg.actionScale));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/decimation", controlCfg.decimation));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_observations", robotCfg_.clipObs));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_actions", robotCfg_.clipActions));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/lin_vel", obsScales.linVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/ang_vel", obsScales.angVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_pos", obsScales.dofPos));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_vel", obsScales.dofVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/height_measurements", obsScales.heightMeasurements));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/actions_size", actionsSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_size", observationSize_));

  actions_.resize(actionsSize_);
  observations_.resize(observationSize_);

  command_.setZero();
  baseLinVel_.setZero();
  basePosition_.setZero();
  std::vector<scalar_t> defaultJointAngles{robotCfg_.initState.L_HAA_joint, robotCfg_.initState.L_HFE_joint,
                                           robotCfg_.initState.L_KFE_joint, robotCfg_.initState.R_HAA_joint,
                                           robotCfg_.initState.R_HFE_joint, robotCfg_.initState.R_KFE_joint};
  lastActions_.resize(actionsSize_);
  defaultJointAngles_.resize(defaultJointAngles.size());
  for (int i = 0; i < defaultJointAngles_.size(); i++) {
    defaultJointAngles_(i, 0) = defaultJointAngles[i];
  }
  return (error == 0);
}

void BipedController::cmdVelCallback(const geometry_msgs::Twist& msg) {
  command_(0) = msg.linear.x;
  command_(1) = msg.linear.y;
  command_(2) = msg.angular.z;
}

void BipedController::stateUpdateCallback(const nav_msgs::Odometry& msg) {
  baseLinVel_(0) = msg.twist.twist.linear.x;
  baseLinVel_(1) = msg.twist.twist.linear.y;
  baseLinVel_(2) = msg.twist.twist.linear.z;
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::BipedController, controller_interface::ControllerBase)
