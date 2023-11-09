//
// Created by luohx on 23-11-7.
//

#include <numeric>

#include "legged_rl_controllers/ParkourController.h"

#include <pluginlib/class_list_macros.hpp>

namespace legged {

void ParkourController::handleWalkMode() {
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
}

bool ParkourController::loadModel(ros::NodeHandle& nh) {
  // create env
  onnxEnvPtr_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
  // create session
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetInterOpNumThreads(1);
  Ort::AllocatorWithDefaultOptions allocator;

  ROS_INFO_STREAM("------- load parkour model -------");
  std::string policyFilePath;
  if (!nh.getParam("/policyFile", policyFilePath)) {
    ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
    return false;
  }
  policyFilePath_ = policyFilePath;
  ROS_INFO_STREAM("Load parkour policy model from path : " << policyFilePath);

  // policy session
  policySessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, policyFilePath.c_str(), sessionOptions);
  policyInputNames_.clear();
  policyOutputNames_.clear();
  policyInputShapes_.clear();
  policyOutputShapes_.clear();
  for (int i = 0; i < policySessionPtr_->GetInputCount(); i++) {
    policyInputNames_.push_back(policySessionPtr_->GetInputName(i, allocator));
    policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (int i = 0; i < policySessionPtr_->GetOutputCount(); i++) {
    policyOutputNames_.push_back(policySessionPtr_->GetOutputName(i, allocator));
    policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load parkour policy model successfully !!!");

  ROS_INFO_STREAM("------- load depth encoder model -------");
  std::string depthEncoderPath;
  if (!nh.getParam("/depthEncoderFile", depthEncoderPath)) {
    ROS_ERROR_STREAM("Get depth encoder path fail from param server, some error occur!");
    return false;
  }
  depthEncoderPath_ = depthEncoderPath;
  ROS_INFO_STREAM("Load depth encoder model from path : " << depthEncoderPath);

  // depth encoder session
  depthEncoderSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, depthEncoderPath_.c_str(), sessionOptions);
  depthEncoderInputNames_.clear();
  depthEncoderOutputNames_.clear();
  depthEncoderInputShapes_.clear();
  depthEncoderOutputShapes_.clear();
  for (int i = 0; i < depthEncoderSessionPtr_->GetInputCount(); i++) {
    depthEncoderInputNames_.push_back(depthEncoderSessionPtr_->GetInputName(i, allocator));
    depthEncoderInputShapes_.push_back(depthEncoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (int i = 0; i < depthEncoderSessionPtr_->GetOutputCount(); i++) {
    depthEncoderOutputNames_.push_back(depthEncoderSessionPtr_->GetOutputName(i, allocator));
    depthEncoderOutputShapes_.push_back(depthEncoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load depth encoder model successfully !!!");
  return true;
}

bool ParkourController::loadRLCfg(ros::NodeHandle& nh) {
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

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_observations", robotCfg_.clipObs));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_actions", robotCfg_.clipActions));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/lin_vel", obsScales.linVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/ang_vel", obsScales.angVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_pos", obsScales.dofPos));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_vel", obsScales.dofVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/height_measurements", obsScales.heightMeasurements));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/actions_size", actionsSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_size", observationsSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/depth_latent_size", depthLatentSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/proprio_observation_size", proprioObservationSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/depth_image/shape", depthShape_));
  combinedImageProPrioSize_ = std::accumulate(depthShape_.begin(), depthShape_.end(), 1, std::multiplies<int>()) + proprioObservationSize_;

  actions_.resize(actionsSize_);
  depthLatent_.resize(depthLatentSize_);
  observations_.resize(observationsSize_ + depthLatentSize_);
  combinedImageProprio_.resize(combinedImageProPrioSize_);

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

void ParkourController::computeActions() {
  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observations_.data(), observations_.size(),
                                                                   policyInputShapes_[0].data(), policyInputShapes_[0].size()));
  // run inference
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      policySessionPtr_->Run(runOptions, policyInputNames_.data(), inputValues.data(), 1, policyOutputNames_.data(), 1);

  for (int i = 0; i < actionsSize_; i++) {
    actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void ParkourController::computeObservation() {
  std::fill(observations_.begin(), observations_.end(), 0);
  std::fill(depthLatent_.begin(), depthLatent_.end(), 0);
  std::fill(combinedImageProprio_.begin(), combinedImageProprio_.end(), 0);
}

void ParkourController::computeDepthLatent() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, combinedImageProprio_.data(), combinedImageProprio_.size(),
                                                                   depthEncoderInputShapes_[0].data(), depthEncoderInputShapes_[0].size()));

  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      depthEncoderSessionPtr_->Run(runOptions, depthEncoderInputNames_.data(), inputValues.data(), 1, depthEncoderOutputNames_.data(), 1);

  for (int i = 0; i < depthLatentSize_; i++) {
    depthLatent_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::ParkourController, controller_interface::ControllerBase)
