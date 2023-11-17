//
// Created by luohx on 23-11-17.
//
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc.hpp>

#include "legged_rl_controllers/BipedVisionController.h"

#include <pluginlib/class_list_macros.hpp>

namespace legged {

void BipedVisionController::update(const ros::Time& time, const ros::Duration& period) {
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

void BipedVisionController::computeActions() {
  // combine observation and depth latent
  for (size_t i = 0; i < observationsSize_; i++) {
    combinedObsDepthLatent_[i] = observations_[i];
  }
  for (size_t i = 0; i < depthLatentSize_; i++) {
    combinedObsDepthLatent_[observationsSize_ + i] = depthLatent_[i];
  }

  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, combinedObsDepthLatent_.data(),
                                                                   combinedObsDepthLatent_.size(), policyInputShapes_[0].data(),
                                                                   policyInputShapes_[0].size()));
  // run inference
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      policySessionPtr_->Run(runOptions, policyInputNames_.data(), inputValues.data(), 1, policyOutputNames_.data(), 1);

  for (size_t i = 0; i < actionsSize_; i++) {
    actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void BipedVisionController::computeObservation() {
  Eigen::Quaternion<scalar_t> quat;
  for (size_t i = 0; i < 4; ++i) {
    quat.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
  }
  vector3_t zyx = quatToZyx(quat);
  matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();

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

  // actions
  vector_t actions(lastActions_);

  // contact states
  vector_t contactFlags(2);
  for (size_t i = 0; i < 2; i++) {
    contactFlags[i] = static_cast<scalar_t>(contactHandles_[i].isContact()) - 0.5;
  }

  // heights
  int sampleCount = 165;
  scalar_t measuredHeight = 0.0;
  scalar_t baseHeight = 0.5;
  vector_t heights(sampleCount);
  heights.fill(baseHeight - 0.3 - measuredHeight);

  RLRobotCfg::ObsScales& obsScales = robotCfg_.obsScales;
  matrix_t commandScaler = Eigen::DiagonalMatrix<scalar_t, 3>(obsScales.linVel, obsScales.linVel, obsScales.angVel);

  vector_t deltaJointPos = jointPos - defaultJointAngles_;

  vector_t proprioObs(proprioObservationSize_);
  // clang-format off
  proprioObs << baseAngVel * obsScales.angVel,
      projectedGravity,
      commandScaler * command,
      deltaJointPos * obsScales.dofPos,
      jointVel * obsScales.dofVel,
      actions,
      contactFlags;
  // clang-format on

  vector_t privExplicit(9);
  privExplicit.setZero();

  vector_t privLatent(17);
  privLatent.setZero();

  if (isFirstRecObs_) {
    proprioObsHistoryBuffer_.resize(proprioObsHistoryLen_ * proprioObservationSize_);
    proprioObsHistoryBuffer_.setZero();
    isFirstRecObs_ = false;
  }

  vector_t obs(observationsSize_);
  obs << proprioObs, heights, privExplicit, privLatent, proprioObsHistoryBuffer_;

  proprioObsHistoryBuffer_.head(proprioObsHistoryBuffer_.size() - proprioObservationSize_) =
      proprioObsHistoryBuffer_.tail(proprioObsHistoryBuffer_.size() - proprioObservationSize_);
  proprioObsHistoryBuffer_.tail(proprioObservationSize_) = proprioObs;

  assert(obs.size() == observationsSize_);
  for (size_t i = 0; i < observationsSize_; i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
  }
  // Limit observation range
  scalar_t obsMin = -robotCfg_.clipObs;
  scalar_t obsMax = robotCfg_.clipObs;
  std::transform(observations_.begin(), observations_.end(), observations_.begin(),
                 [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

void BipedVisionController::computeDepthLatent() {
  const auto& lastImage = depthBufferPtr_->front();
  for (size_t i = 0; i < numPixels_; i++) {
    combinedImageProprio_[i] = lastImage[i];
  }
  for (size_t i = 0; i < proprioObservationSize_; i++) {
    combinedImageProprio_[numPixels_ + i] = observations_[i];
  }

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, combinedImageProprio_.data(), combinedImageProprio_.size(),
                                                                   depthEncoderInputShapes_[0].data(), depthEncoderInputShapes_[0].size()));

  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      depthEncoderSessionPtr_->Run(runOptions, depthEncoderInputNames_.data(), inputValues.data(), 1, depthEncoderOutputNames_.data(), 1);

  for (size_t i = 0; i < depthLatentSize_; i++) {
    depthLatent_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

bool BipedVisionController::loadModel(ros::NodeHandle& nh) {
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
  for (size_t i = 0; i < policySessionPtr_->GetInputCount(); i++) {
    policyInputNames_.push_back(policySessionPtr_->GetInputName(i, allocator));
    policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < policySessionPtr_->GetOutputCount(); i++) {
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
  depthEncoderInputNames_.clear();
  depthEncoderOutputNames_.clear();
  depthEncoderInputShapes_.clear();
  depthEncoderOutputShapes_.clear();
  depthEncoderSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, depthEncoderPath_.c_str(), sessionOptions);
  for (size_t i = 0; i < depthEncoderSessionPtr_->GetInputCount(); i++) {
    depthEncoderInputNames_.push_back(depthEncoderSessionPtr_->GetInputName(i, allocator));
    depthEncoderInputShapes_.push_back(depthEncoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < depthEncoderSessionPtr_->GetOutputCount(); i++) {
    depthEncoderOutputNames_.push_back(depthEncoderSessionPtr_->GetOutputName(i, allocator));
    depthEncoderOutputShapes_.push_back(depthEncoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load depth encoder model successfully !!!");
  return true;
}

bool BipedVisionController::loadRLCfg(ros::NodeHandle& nh) {
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
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_size", observationsSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/depth_latent_size", depthLatentSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/proprio_observation_size", proprioObservationSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/proprio_history_len", proprioObsHistoryLen_));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/depth_image/original", depthOriginalShape_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/depth_image/resized", depthResizedShape_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/depth_image/near_clip", nearClip_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/depth_image/far_clip", farClip_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/depth_image/buffer_len", depthBufferLen_));

  actions_.resize(actionsSize_);
  depthLatent_.resize(depthLatentSize_);
  observations_.resize(observationsSize_);
  numPixels_ = depthResizedShape_[0] * depthResizedShape_[1];
  combinedImageProPrioSize_ = numPixels_ + proprioObservationSize_;
  combinedImageProprio_.resize(combinedImageProPrioSize_);
  combinedObsDepthLatent_.resize(observationsSize_ + depthLatentSize_);

  command_.setZero();
  lastActions_.resize(actionsSize_);
  std::vector<scalar_t> defaultJointAngles{robotCfg_.initState.L_HAA_joint, robotCfg_.initState.L_HFE_joint,
                                           robotCfg_.initState.L_KFE_joint, robotCfg_.initState.R_HAA_joint,
                                           robotCfg_.initState.R_HFE_joint, robotCfg_.initState.R_KFE_joint};
  defaultJointAngles_.resize(defaultJointAngles.size());

  for (size_t i = 0; i < defaultJointAngles.size(); i++) {
    defaultJointAngles_(i, 0) = defaultJointAngles[i];
  }

  resizedDepthImagePub_ = nh.advertise<sensor_msgs::Image>("/d435F/aligned_depth_to_color/image_resized", 1, true);
  depthImageSub_ = nh.subscribe("/d435F/aligned_depth_to_color/image_raw", 1, &BipedVisionController::depthImageCallback, this);

  return (error == 0);
}

void BipedVisionController::depthImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
  uint32_t imageWidth = msg->width;
  uint32_t imageHeight = msg->height;
  std::string encoding = msg->encoding;

  std::vector<float> imageData;
  uint32_t imageNumPixel = imageWidth * imageHeight;
  imageData.reserve(imageNumPixel);

  for (size_t i = 0; i < imageNumPixel; i++) {
    uint16_t pixelValue = (msg->data[i * 2 + 1] << 8) | msg->data[i * 2];
    float distance = -static_cast<float>(pixelValue) / 1000;
    distance = std::min(std::max(distance, -farClip_), -nearClip_);
    imageData.push_back(distance);
  }

  std::vector<float> cropImage = cropDepthImage(imageData, imageWidth, imageHeight, 4, 4, 0, 2);

  cv::Mat srcImage(imageHeight - 2, imageWidth - 8, CV_32F, cropImage.data());
  cv::Size targetResize(depthResizedShape_[0], depthResizedShape_[1]);
  cv::Mat resizedImage;
  cv::resize(srcImage, resizedImage, targetResize, 0, 0, cv::INTER_CUBIC);
  std::vector<float> resizedImageData(resizedImage.begin<float>(), resizedImage.end<float>());

  for (size_t i = 0; i < resizedImageData.size(); i++) {
    resizedImageData[i] *= -1;
    resizedImageData[i] = (resizedImageData[i] - nearClip_) / (farClip_ - nearClip_) - 0.5;
  }

  sensor_msgs::Image resizeImgMsg;
  resizeImgMsg.step = msg->step;
  resizeImgMsg.encoding = msg->encoding;
  resizeImgMsg.header.stamp = ros::Time::now();
  resizeImgMsg.header.frame_id = msg->header.frame_id;
  resizeImgMsg.width = depthResizedShape_[0];
  resizeImgMsg.height = depthResizedShape_[1];
  resizeImgMsg.is_bigendian = msg->is_bigendian;
  resizeImgMsg.data.resize(depthResizedShape_[0] * depthResizedShape_[1] * 2);
  for (size_t i = 0; i < resizedImageData.size(); i++) {
    auto distance = static_cast<uint16_t>((resizedImageData[i] + 0.5) * 1000);
    resizeImgMsg.data[i * 2] = distance & 0xFF;
    resizeImgMsg.data[i * 2 + 1] = distance >> 8;
  }
  resizedDepthImagePub_.publish(resizeImgMsg);

  if (isFirstRecDepth_) {
    depthBufferPtr_ = std::make_shared<std::deque<std::vector<float>>>();
    for (size_t i = 0; i < depthBufferLen_; i++) {
      depthBufferPtr_->push_back(resizedImageData);
    }
    isFirstRecDepth_ = false;
  } else {
    if (!depthBufferPtr_->empty()) {
      depthBufferPtr_->pop_front();
      depthBufferPtr_->push_back(resizedImageData);
    } else {
      ROS_ERROR("depth buffer is empty, could not be updated");
    }
  }

  this->computeDepthLatent();
}

std::vector<float> BipedVisionController::cropDepthImage(const std::vector<float>& image, int width, int height, int left, int right,
                                                         int top, int bottom) {
  if (image.empty() || width <= 0 || height <= 0) {
    return {};
  }

  int cropped_width = width - left - right;
  int cropped_height = height - top - bottom;

  if (cropped_width <= 0 || cropped_height <= 0) {
    return {};
  }

  std::vector<float> cropped_image(cropped_width * cropped_height);

  for (int i = 0; i < cropped_height; ++i) {
    std::copy(image.begin() + (i + top) * width + left, image.begin() + (i + top) * width + left + cropped_width,
              cropped_image.begin() + i * cropped_width);
  }

  return cropped_image;
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::BipedVisionController, controller_interface::ControllerBase)
