//
// Created by luohx on 23-8-23.
//

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <pinocchio/algorithm/jacobian.hpp>

#include "legged_rl_controllers/StudentPolicyController.h"

#include <pluginlib/class_list_macros.hpp>

namespace legged {

bool StudentPolicyController::init(hardware_interface::RobotHW* robotHw, ros::NodeHandle& controllerNH) {
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
  std::string policyFileDir;
  if (!controllerNH.getParam("/module/module_dir", policyFileDir)) {
    ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
    return false;
  }
  loadModel(policyFileDir);

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
  //  baseStateSub_ = controllerNH.subscribe("/gazebo/model_states", 1, &LeggedRLController::baseStateRecCallback, this);
  cmdVelSub_ = controllerNH.subscribe("/cmd_vel", 1, &StudentPolicyController::cmdVelCallback, this);
  return true;
}

void StudentPolicyController::update(const ros::Time& time, const ros::Duration& period) {
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

void StudentPolicyController::loadModel(const std::string& policyFileDir) {
  std::cerr << "load student policy model" << std::endl;
  std::string policyModelPath = policyFileDir + "policy_1.onnx";
  std::string encoderModelPath = policyFileDir + "blind_encoder.onnx";

  // create env
  onnxEnvPrt_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
  // create session
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(1);
  sessionOptions.SetInterOpNumThreads(1);

  Ort::AllocatorWithDefaultOptions allocator;
  // policy session
  policySessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyModelPath.c_str(), sessionOptions);
  policyInputNames_.clear();
  policyOutputNames_.clear();
  policyInputShapes_.clear();
  policyOutputShapes_.clear();
  for (int i = 0; i < policySessionPtr_->GetInputCount(); i++) {
    policyInputNames_.push_back(policySessionPtr_->GetInputName(i, allocator));
    policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << policySessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < policySessionPtr_->GetOutputCount(); i++) {
    policyOutputNames_.push_back(policySessionPtr_->GetOutputName(i, allocator));
    policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  // encoder session
  encoderSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, encoderModelPath.c_str(), sessionOptions);
  encoderInputNames_.clear();
  encoderOutputNames_.clear();
  encoderInputShapes_.clear();
  encoderOutputShapes_.clear();
  for (int i = 0; i < encoderSessionPtr_->GetInputCount(); i++) {
    encoderInputNames_.push_back(encoderSessionPtr_->GetInputName(i, allocator));
    encoderInputShapes_.push_back(encoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << encoderSessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = encoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    encoderBatchSize_ = shape[0];
    encoderSeqLength_ = shape[1];
    encoderInputDim_ = shape[2];
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < encoderSessionPtr_->GetOutputCount(); i++) {
    encoderOutputNames_.push_back(encoderSessionPtr_->GetOutputName(i, allocator));
    std::cerr << encoderSessionPtr_->GetOutputName(i, allocator) << std::endl;
    encoderOutputShapes_.push_back(encoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = encoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  ROS_INFO_STREAM("Load Onnx model successfully !!!");
}

void StudentPolicyController::computeActions() {
  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observations_.data(), observations_.size(),
                                                                   policyInputShapes_[0].data(), policyInputShapes_[0].size()));
  // run inference
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      policySessionPtr_->Run(runOptions, policyInputNames_.data(), inputValues.data(), 1, policyOutputNames_.data(), 1);

  for (int i = 0; i < 12; i++) {
    actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void StudentPolicyController::computeObservation(const ros::Time& time, const ros::Duration& period) {
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

  vector_t proprioObs(encoderInputDim_);
  // clang-format off
  proprioObs << baseLinVel * obsScales.linVel,
      baseAngVel * obsScales.angVel,
      projectedGravity,
      commandScaler * command,
      (jointPos - defaultJointAngles_) * obsScales.dofPos,
      jointVel * obsScales.dofVel,
      actions;
  // clang-format on

  if (isfirstRecObs_) {
    int64_t inputSize =
        std::accumulate(encoderInputShapes_[0].begin(), encoderInputShapes_[0].end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    proprioHistoryBuffer_.resize(inputSize);
    for (size_t i = 0; i < encoderSeqLength_; i++) {
      proprioHistoryBuffer_.segment(i * encoderInputDim_, encoderInputDim_) = proprioObs.cast<tensor_element_t>();
    }
    isfirstRecObs_ = false;
  }
  proprioHistoryBuffer_.head(proprioHistoryBuffer_.size() - encoderInputDim_) =
      proprioHistoryBuffer_.tail(proprioHistoryBuffer_.size() - encoderInputDim_);
  proprioHistoryBuffer_.tail(encoderInputDim_) = proprioObs.cast<tensor_element_t>();

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, proprioHistoryBuffer_.data(), proprioHistoryBuffer_.size(),
                                                                   encoderInputShapes_[0].data(), encoderInputShapes_[0].size()));
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      encoderSessionPtr_->Run(runOptions, encoderInputNames_.data(), inputValues.data(), 1, encoderOutputNames_.data(), 1);

  vector_t estimatedHeights(96);
  for (int i = 0; i < 96; i++) {
    estimatedHeights(i) = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }

  vector_t obs(proprioObs.size() + estimatedHeights.size());
  obs << proprioObs, estimatedHeights;

  for (size_t i = 0; i < obs.size(); i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
  }
  // Limit observation range
  scalar_t obsMin = -robotCfg_.clipObs;
  scalar_t obsMax = robotCfg_.clipObs;
  std::transform(observations_.begin(), observations_.end(), observations_.begin(),
                 [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

void StudentPolicyController::cmdVelCallback(const geometry_msgs::Twist& msg) {
  LeggedRLController::cmdVelCallback(msg);
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::StudentPolicyController, controller_interface::ControllerBase)
