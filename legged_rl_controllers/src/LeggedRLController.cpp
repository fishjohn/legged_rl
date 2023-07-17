//
// Created by luohx on 22-12-5.
//

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <pinocchio/algorithm/jacobian.hpp>

#include "legged_rl_controllers/LeggedRLController.h"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>

#include <angles/angles.h>
#include <pluginlib/class_list_macros.hpp>

namespace legged {
LeggedRLController::~LeggedRLController() {
  controllerRunning_ = false;
}

bool LeggedRLController::init(hardware_interface::RobotHW* robotHw, ros::NodeHandle& controllerNH) {
  // Get config file and setup legged interface
  std::string taskFile;
  std::string urdfFile;
  std::string referenceFile;
  controllerNH.getParam("/urdfFile", urdfFile);
  controllerNH.getParam("/taskFile", taskFile);
  controllerNH.getParam("/referenceFile", referenceFile);
  bool verbose = false;
  loadData::loadCppDataType(taskFile, "legged_robot_interface.verbose", verbose);

  setupLeggedInterface(taskFile, urdfFile, referenceFile, verbose);
  CentroidalModelPinocchioMapping pinocchioMapping(leggedInterface_->getCentroidalModelInfo());
  eeKinematicsPtr_ = std::make_shared<PinocchioEndEffectorKinematics>(leggedInterface_->getPinocchioInterface(), pinocchioMapping,
                                                                      leggedInterface_->modelSettings().contactNames3DoF);
  rbdConversions_ = std::make_shared<CentroidalModelRbdConversions>(leggedInterface_->getPinocchioInterface(),
                                                                    leggedInterface_->getCentroidalModelInfo());

  // Get policy model
  std::string policyFilePath;
  if (!controllerNH.getParam("/module/module_path", policyFilePath)) {
    ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
    return false;
  }
  loadPolicyModel(policyFilePath);

  // Get robot config from param server
  if (!this->parseCfg(controllerNH)) {
    ROS_ERROR_STREAM("Get robot config fail from param server, some error occur!");
    return false;
  }
  actions_.resize(actionsSize_);
  observations_.resize(observationSize_);

  // State init
  command_.setZero();
  baseLinVel_.setZero();
  basePosition_.setZero();
  std::vector<scalar_t> defaultJointAngles{
      robotCfg_.initState.LF_HAA_joint, robotCfg_.initState.LF_HFE_joint, robotCfg_.initState.LF_KFE_joint,
      robotCfg_.initState.RF_HAA_joint, robotCfg_.initState.RF_HFE_joint, robotCfg_.initState.RF_KFE_joint,
      robotCfg_.initState.LH_HAA_joint, robotCfg_.initState.LH_HFE_joint, robotCfg_.initState.LH_KFE_joint,
      robotCfg_.initState.RH_HAA_joint, robotCfg_.initState.RH_HFE_joint, robotCfg_.initState.RH_KFE_joint};
  lastActions_.resize(leggedInterface_->getCentroidalModelInfo().actuatedDofNum);
  defaultJointAngles_.resize(leggedInterface_->getCentroidalModelInfo().actuatedDofNum);
  for (int i = 0; i < leggedInterface_->getCentroidalModelInfo().actuatedDofNum; i++) {
    defaultJointAngles_(i, 0) = defaultJointAngles[i];
  }

  // Hardware interface
  auto* hybridJointInterface = robotHw->get<HybridJointInterface>();
  const auto& jointNames = leggedInterface_->modelSettings().jointNames;
  for (const auto& jointName : jointNames) {
    hybridJointHandles_.push_back(hybridJointInterface->getHandle(jointName));
  }
  imuSensorHandles_ = robotHw->get<hardware_interface::ImuSensorInterface>()->getHandle("unitree_imu");

  auto* contactInterface = robotHw->get<ContactSensorInterface>();
  const auto& footNames = leggedInterface_->modelSettings().contactNames3DoF;
  for (const auto& footName : footNames) {
    contactHandles_.push_back(contactInterface->getHandle(footName));
  }

  // State estimate
  setupStateEstimate(taskFile, verbose);

  // Init publisher and subscriber
  baseStateSub_ = controllerNH.subscribe("/gazebo/model_states", 1, &LeggedRLController::baseStateRecCallback, this);
  cmdVelSub_ = controllerNH.subscribe("/cmd_vel", 1, &LeggedRLController::cmdVelCallback, this);
  return true;
}

void LeggedRLController::starting(const ros::Time& time) {
  // Initial state
  //  currentObservation_.state.setZero(leggedInterface_->getCentroidalModelInfo().stateDim);
  updateStateEstimation(time, ros::Duration(0.002));
  //  currentObservation_.input.setZero(leggedInterface_->getCentroidalModelInfo().inputDim);
  //  currentObservation_.mode = ModeNumber::STANCE;

  controllerRunning_ = true;

  std::vector<scalar_t> initJointAngles(12, 0);
  for (auto& hybridJointHandle : hybridJointHandles_) {
    initJointAngles_.push_back(hybridJointHandle.getPosition());
  }

  scalar_t durationSecs = 2.0;
  standDuration_ = durationSecs * 1000.0;
  standPercent_ += 1 / standDuration_;
  mode_ = Mode::LIE;
  loopCount_ = 0;
}

void LeggedRLController::stopping(const ros::Time& time) {
  controllerRunning_ = false;
}

void LeggedRLController::update(const ros::Time& time, const ros::Duration& period) {
  //  // state estimate
  updateStateEstimation(time, period);

  if (mode_ == Mode::LIE) {
    if (standPercent_ < 1) {
      for (int j = 0; j < hybridJointHandles_.size(); j++) {
        scalar_t pos_des = initJointAngles_[j] * (1 - standPercent_) + defaultJointAngles_[j] * standPercent_;
        hybridJointHandles_[j].setCommand(pos_des, 0, 50, 1, 0);
      }
      standPercent_ += 1 / standDuration_;
    } else {
      mode_ = Mode::STAND;
    }
  } else if (mode_ == Mode::STAND) {
    if (loopCount_ > 5000) {
      mode_ = Mode::WALK;
    }
  } else if (mode_ == Mode::WALK) {
    // compute observation & actions
    if (loopCount_ % robotCfg_.controlCfg.decimation == 0) {
      computeObservation(time, period);
      computeActions();
      // limit action range
      scalar_t action_min = -robotCfg_.clipActions;
      scalar_t action_max = robotCfg_.clipActions;
      std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                     [action_min, action_max](scalar_t x) { return std::max(action_min, std::min(action_max, x)); });
    }

    // set action
    for (int i = 0; i < hybridJointHandles_.size(); i++) {
      scalar_t pos_des = actions_[i] * robotCfg_.controlCfg.actionScale + defaultJointAngles_(i, 0);
      hybridJointHandles_[i].setCommand(pos_des, 0, robotCfg_.controlCfg.stiffness, robotCfg_.controlCfg.damping, 0);
      lastActions_(i, 0) = actions_[i];
    }
  }
  loopCount_++;
}

void LeggedRLController::updateStateEstimation(const ros::Time& time, const ros::Duration& period) {
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  contact_flag_t contacts;
  Eigen::Quaternion<scalar_t> quat;
  contact_flag_t contactFlag;
  vector3_t angularVel, linearAccel;
  matrix3_t orientationCovariance, angularVelCovariance, linearAccelCovariance;

  for (size_t i = 0; i < hybridJointHandles_.size(); ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }
  for (size_t i = 0; i < contacts.size(); ++i) {
    contactFlag[i] = contactHandles_[i].isContact();
  }
  for (size_t i = 0; i < 4; ++i) {
    quat.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
  }
  for (size_t i = 0; i < 3; ++i) {
    angularVel(i) = imuSensorHandles_.getAngularVelocity()[i];
    linearAccel(i) = imuSensorHandles_.getLinearAcceleration()[i];
  }
  for (size_t i = 0; i < 9; ++i) {
    orientationCovariance(i) = imuSensorHandles_.getOrientationCovariance()[i];
    angularVelCovariance(i) = imuSensorHandles_.getAngularVelocityCovariance()[i];
    linearAccelCovariance(i) = imuSensorHandles_.getLinearAccelerationCovariance()[i];
  }

  stateEstimate_->updateJointStates(jointPos, jointVel);
  stateEstimate_->updateContact(contactFlag);
  stateEstimate_->updateImu(quat, angularVel, linearAccel, orientationCovariance, angularVelCovariance, linearAccelCovariance);
  rbdState_ = stateEstimate_->update(time, period);
  //  currentObservation_.time += period.toSec();
  //  scalar_t yawLast = currentObservation_.state(9);
  //  currentObservation_.state = rbdConversions_->computeCentroidalStateFromRbdModel(measuredRbdState_);
  //  currentObservation_.state(9) = yawLast + angles::shortest_angular_distance(yawLast, currentObservation_.state(9));
  //  currentObservation_.mode = stateEstimate_->getMode();
}

void LeggedRLController::setupLeggedInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile,
                                              bool verbose) {
  leggedInterface_ = std::make_shared<LeggedInterface>(taskFile, urdfFile, referenceFile);
  leggedInterface_->setupOptimalControlProblem(taskFile, urdfFile, referenceFile, verbose);
}

void LeggedRLController::setupStateEstimate(const std::string& taskFile, bool verbose) {
  stateEstimate_ = std::make_shared<KalmanFilterEstimate>(leggedInterface_->getPinocchioInterface(),
                                                          leggedInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_);
  dynamic_cast<KalmanFilterEstimate&>(*stateEstimate_).loadSettings(taskFile, verbose);
}

void LeggedRLController::loadPolicyModel(const std::string& policy_file_path) {
  policyFilePath_ = policy_file_path;
  ROS_INFO_STREAM("Load Onnx model from path : " << policy_file_path);

  // create env
  onnxEnvPrt_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
  // create session
  Ort::SessionOptions session_options;
  session_options.SetInterOpNumThreads(1);
  sessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policy_file_path.c_str(), session_options);
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
}

void LeggedRLController::computeActions() {
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

void LeggedRLController::computeObservation(const ros::Time& time, const ros::Duration& period) {
  const auto& info = leggedInterface_->getCentroidalModelInfo();

  vector3_t zyx = rbdState_.segment<3>(0);
  matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();

  // linear velocity (base frame)
  vector3_t baseLinVel = inverseRot * rbdState_.segment<3>(info.generalizedCoordinatesNum + 3);
  // Angular velocity
  vector3_t baseAngVel(imuSensorHandles_.getAngularVelocity()[0], imuSensorHandles_.getAngularVelocity()[1],
                       imuSensorHandles_.getAngularVelocity()[2]);

  // Projected gravity
  vector3_t gravityVector(0, 0, -1);
  vector3_t projectedGravity(inverseRot * gravityVector);

  // command
  vector3_t command = command_;

  // dof position and dof velocity
  vector_t jointPos = rbdState_.segment(6, info.actuatedDofNum);
  vector_t jointVel = rbdState_.segment(6 + info.generalizedCoordinatesNum, info.actuatedDofNum);

  // actions
  vector_t actions(lastActions_);

  // heights
  int sampleCount = 187;
  scalar_t measuredHeight = 0.0;
  scalar_t baseHeight = rbdState_(5);
  vector_t heights(sampleCount);
  heights.fill(baseHeight - 0.5 - measuredHeight);

  ObsScales& obsScales = robotCfg_.obsScales;
  matrix_t commandScaler = Eigen::DiagonalMatrix<scalar_t, 3>(obsScales.linVel, obsScales.linVel, obsScales.angVel);

  vector_t obs(235);
  // clang-format off
  obs << baseLinVel * obsScales.linVel,
      baseAngVel * obsScales.angVel,
      projectedGravity,
      commandScaler * command,
      (jointPos - defaultJointAngles_) * obsScales.dofPos,
      jointVel * obsScales.dofVel,
      actions,
      heights * obsScales.heightMeasurements;
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

void LeggedRLController::baseStateRecCallback(const gazebo_msgs::ModelStates& msg) {
  baseLinVel_.x() = msg.twist[1].linear.x;
  baseLinVel_.y() = msg.twist[1].linear.y;
  baseLinVel_.z() = msg.twist[1].linear.z;

  basePosition_.x() = msg.pose[1].position.x;
  basePosition_.y() = msg.pose[1].position.y;
  basePosition_.z() = msg.pose[1].position.z;
}

void LeggedRLController::cmdVelCallback(const geometry_msgs::Twist& msg) {
  command_[0] = msg.linear.x;
  command_[1] = msg.linear.y;
  command_[2] = msg.linear.z;
}

bool LeggedRLController::parseCfg(ros::NodeHandle& nh) {
  InitState& initState = robotCfg_.initState;
  ControlCfg& controlCfg = robotCfg_.controlCfg;
  ObsScales& obsScales = robotCfg_.obsScales;

  int error = 0;
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/LF_HAA_joint", initState.LF_HAA_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/LF_HFE_joint", initState.LF_HFE_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/LF_KFE_joint", initState.LF_KFE_joint));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/RF_HAA_joint", initState.RF_HAA_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/RF_HFE_joint", initState.RF_HFE_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/RF_KFE_joint", initState.RF_KFE_joint));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/LH_HAA_joint", initState.LH_HAA_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/LH_HFE_joint", initState.LH_HFE_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/LH_KFE_joint", initState.LH_KFE_joint));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/RH_HAA_joint", initState.RH_HAA_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/RH_HFE_joint", initState.RH_HFE_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/RH_KFE_joint", initState.RH_KFE_joint));

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

  return (error == 0);
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::LeggedRLController, controller_interface::ControllerBase)
