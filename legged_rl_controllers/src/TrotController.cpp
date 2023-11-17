//
// Created by luohx on 23-8-29.
//

#include <pinocchio/fwd.hpp>

// forward declarations must be included first.
#include <sys/time.h>
#include <pluginlib/class_list_macros.hpp>
#include "legged_rl_controllers/TrotController.h"

namespace legged {

void TrotController::handleWalkMode() {
  // compute observation & actions
  if (loopCount_ % robotCfg_.controlCfg.decimation == 0) {
    computeObservation();
    computeEncoder();
    computeActions();
    // limit action range
    scalar_t actionMin = -robotCfg_.clipActions;
    scalar_t actionMax = robotCfg_.clipActions;
    std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                   [actionMin, actionMax](scalar_t x) { return std::max(actionMin, std::min(actionMax, x)); });
  }

  // set action
  const auto& info = leggedInterface_->getCentroidalModelInfo();
  vector_t jointPos = rbdState_.segment(6, info.actuatedDofNum);
  vector_t jointVel = rbdState_.segment(6 + info.generalizedCoordinatesNum, info.actuatedDofNum);
  for (int i = 0; i < hybridJointHandles_.size(); i++) {
    scalar_t actionMin =
        jointPos(i) - defaultJointAngles_(i, 0) +
        (robotCfg_.controlCfg.damping * jointVel(i) - robotCfg_.controlCfg.user_torque_limit) / robotCfg_.controlCfg.stiffness;
    scalar_t actionMax =
        jointPos(i) - defaultJointAngles_(i, 0) +
        (robotCfg_.controlCfg.damping * jointVel(i) + robotCfg_.controlCfg.user_torque_limit) / robotCfg_.controlCfg.stiffness;
    // scalar_t actionMin = jointPos(i) - defaultJointAngles_(i, 0) - robotCfg_.controlCfg.user_torque_limit /
    // robotCfg_.controlCfg.stiffness; scalar_t actionMax = jointPos(i) - defaultJointAngles_(i, 0) + robotCfg_.controlCfg.user_torque_limit
    // / robotCfg_.controlCfg.stiffness;
    actions_[i] = std::max(actionMin, std::min(actionMax, (scalar_t)actions_[i]));
    scalar_t pos_des = actions_[i] * robotCfg_.controlCfg.actionScale + defaultJointAngles_(i, 0);
    hybridJointHandles_[i].setCommand(pos_des, 0, robotCfg_.controlCfg.stiffness, robotCfg_.controlCfg.damping, 0);
    lastActions_(i, 0) = actions_[i];
  }
}

bool TrotController::loadModel(ros::NodeHandle& nh) {
  ROS_INFO_STREAM("load policy model");

  std::string policyModelPath;
  std::string encoderModelPath;
  std::string gaitGeneratorModelPath;
  if (!nh.getParam("/policyModelPath", policyModelPath) || !nh.getParam("/encoderModelPath", encoderModelPath) ||
      !nh.getParam("/gaitGeneratorModelPath", gaitGeneratorModelPath)) {
    ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
    return false;
  }

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
    std::cerr << policySessionPtr_->GetOutputName(i, allocator) << std::endl;
    policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
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

  // gait generator session
  gaitGeneratorSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, gaitGeneratorModelPath.c_str(), sessionOptions);
  gaitGeneratorInputNames_.clear();
  gaitGeneratorOutputNames_.clear();
  gaitGeneratorInputShapes_.clear();
  gaitGeneratorOutputShapes_.clear();
  for (int i = 0; i < gaitGeneratorSessionPtr_->GetInputCount(); i++) {
    gaitGeneratorInputNames_.push_back(gaitGeneratorSessionPtr_->GetInputName(i, allocator));
    gaitGeneratorInputShapes_.push_back(gaitGeneratorSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << gaitGeneratorSessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = gaitGeneratorSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < gaitGeneratorSessionPtr_->GetOutputCount(); i++) {
    gaitGeneratorOutputNames_.push_back(gaitGeneratorSessionPtr_->GetOutputName(i, allocator));
    std::cerr << gaitGeneratorSessionPtr_->GetOutputName(i, allocator) << std::endl;
    gaitGeneratorOutputShapes_.push_back(gaitGeneratorSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = gaitGeneratorSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
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
  return true;
}

bool TrotController::loadRLCfg(ros::NodeHandle& nh) {
  RLRobotCfg::InitState& initState = robotCfg_.initState;
  RLRobotCfg::ControlCfg& controlCfg = robotCfg_.controlCfg;
  RLRobotCfg::ObsScales& obsScales = robotCfg_.obsScales;

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
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/user_torque_limit", controlCfg.user_torque_limit));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_observations", robotCfg_.clipObs));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_actions", robotCfg_.clipActions));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/lin_vel", obsScales.linVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/ang_vel", obsScales.angVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_pos", obsScales.dofPos));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_vel", obsScales.dofVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/height_measurements", obsScales.heightMeasurements));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/actions_size", actionsSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_size", observationSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/commands_size", commandsSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/obs_history_length", obsHistoryLength_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/encoder_output_size", encoderOutputSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/gait_generator_output_size", gaitGeneratorOutputSize_));
  encoderIntputSize_ = obsHistoryLength_ * observationSize_;

  actions_.resize(actionsSize_);
  observations_.resize(observationSize_);
  commands_.resize(commandsSize_);
  encoderOut_.resize(encoderOutputSize_);
  gaitGeneratorOut_.resize(gaitGeneratorOutputSize_);

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

  return (error == 0);
}

void TrotController::computeActions() {
  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  std::vector<tensor_element_t> combined_obs;
  for (const auto& item : encoderOut_) {
    combined_obs.push_back(item);
  }
  for (const auto& item : observations_) {
    combined_obs.push_back(item);
  }
  for (const auto& item : commands_) {
    combined_obs.push_back(item);
  }
  for (const auto& item : gaitGeneratorOut_) {
    combined_obs.push_back(item);
  }
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, combined_obs.data(), combined_obs.size(),
                                                                   policyInputShapes_[0].data(), policyInputShapes_[0].size()));
  // run inference
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      policySessionPtr_->Run(runOptions, policyInputNames_.data(), inputValues.data(), 1, policyOutputNames_.data(), 1);

  for (int i = 0; i < actionsSize_; i++) {
    actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void TrotController::computeEncoder() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, proprioHistoryBuffer_.data(), proprioHistoryBuffer_.size(),
                                                                   encoderInputShapes_[0].data(), encoderInputShapes_[0].size()));
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      encoderSessionPtr_->Run(runOptions, encoderInputNames_.data(), inputValues.data(), 1, encoderOutputNames_.data(), 1);

  for (int i = 0; i < encoderOutputSize_; i++) {
    encoderOut_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }

  Ort::MemoryInfo gaitGeneratorMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> gaitGeneratorInputValues;
  std::vector<tensor_element_t> combined_input;
  for (const auto& item : observations_) {
    combined_input.push_back(item);
  }
  for (const auto& item : commands_) {
    combined_input.push_back(item);
  }
  gaitGeneratorInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(gaitGeneratorMemoryInfo, combined_input.data(),
                                                                                combined_input.size(), gaitGeneratorInputShapes_[0].data(),
                                                                                gaitGeneratorInputShapes_[0].size()));
  Ort::RunOptions gaitGeneratorRunOptions;
  std::vector<Ort::Value> gaitGeneratorOutputValues = gaitGeneratorSessionPtr_->Run(
      gaitGeneratorRunOptions, gaitGeneratorInputNames_.data(), gaitGeneratorInputValues.data(), 1, gaitGeneratorOutputNames_.data(), 1);

  tensor_element_t gaitGeneratorOutNorm = 0;
  for (int i = 0; i < gaitGeneratorOutputSize_; i++) {
    gaitGeneratorOut_[i] = *(gaitGeneratorOutputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    gaitGeneratorOutNorm += gaitGeneratorOut_[i] * gaitGeneratorOut_[i];
  }
  for (int i = 0; i < gaitGeneratorOutputSize_; i++) {
    gaitGeneratorOut_[i] /= sqrt(gaitGeneratorOutNorm);
  }
}

void TrotController::computeObservation() {
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

  vector_t gait(7);
  gait << 2.5, 0.5, 0.5, 0.0, 0.5, 0.08, 0.3;  // trot
  //   gait << 2.5, 0.5, 0.25, 0.75, 0.75, 0.08, 0.3;  // walking
  //   gait << 2.5, 0.5, 0.0, 0.5, 0.5, 0.08, 0.3;  // pacing
  //   gait << 2.5, 0.0, 0.0, 0.0, 0.5, 0.08, 0.3;  // pronking
  gait_index_ += 0.02 * gait(0);
  if (gait_index_ > 1.0) {
    gait_index_ = 0.0;
  }
  //   if (fabsf(command[0]) + fabsf(command[1]) + fabsf(command[2]) < 0.01) {
  //     gait_index_ = 0.0;
  //   }

  vector_t gait_clock(2);
  gait_clock << sin(gait_index_ * 2 * M_PI), cos(gait_index_ * 2 * M_PI);

  // actions
  vector_t actions(lastActions_);

  // heights
  //  int sampleCount = 187;
  //  scalar_t measuredHeight = 0.0;
  //  scalar_t baseHeight = rbdState_(5);
  //  vector_t heights(sampleCount);
  //  heights.fill(baseHeight - 0.5 - measuredHeight);

  RLRobotCfg::ObsScales& obsScales = robotCfg_.obsScales;
  matrix_t commandScaler = Eigen::DiagonalMatrix<scalar_t, 3>(obsScales.linVel, obsScales.linVel, obsScales.angVel);
  vector_t obs(observationSize_);

  obs << baseAngVel * obsScales.angVel,                     //
      projectedGravity,                                     //
      (jointPos - defaultJointAngles_) * obsScales.dofPos,  //
      jointVel * obsScales.dofVel,                          //
      actions;

  command = commandScaler * command;

  if (isfirstRecObs_) {
    int64_t inputSize =
        std::accumulate(encoderInputShapes_[0].begin(), encoderInputShapes_[0].end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    proprioHistoryBuffer_.resize(inputSize);
    for (size_t i = 0; i < obsHistoryLength_; i++) {
      proprioHistoryBuffer_.segment(i * observationSize_, observationSize_) = obs.cast<tensor_element_t>();
    }
    isfirstRecObs_ = false;
  }
  proprioHistoryBuffer_.head(proprioHistoryBuffer_.size() - observationSize_) =
      proprioHistoryBuffer_.tail(proprioHistoryBuffer_.size() - observationSize_);
  proprioHistoryBuffer_.tail(observationSize_) = obs.cast<tensor_element_t>();

  // clang-format on
  //   printf("observation\n");
  for (size_t i = 0; i < obs.size(); i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
    // std::cout << observations_[i] << std::endl;
  }
  for (size_t i = 0; i < command.size(); i++) {
    commands_[i] = static_cast<tensor_element_t>(command(i));
    // std::cout << commands_[i] << std::endl;
  }
  // Limit observation range
  scalar_t obsMin = -robotCfg_.clipObs;
  scalar_t obsMax = robotCfg_.clipObs;

  std::transform(observations_.begin(), observations_.end(), observations_.begin(),
                 [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::TrotController, controller_interface::ControllerBase)
