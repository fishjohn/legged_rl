//
// Created by luohx on 23-11-17.
//

#pragma once

#include "legged_rl_controllers/BipedController.h"

#include <deque>

namespace legged {
using namespace ocs2;
using namespace legged_robot;

class BipedVisionController : public BipedController {
  using tensor_element_t = float;

 public:
  BipedVisionController() = default;
  virtual ~BipedVisionController() = default;
  void update(const ros::Time& time, const ros::Duration& period) override;

 protected:
  bool loadModel(ros::NodeHandle& nh) override;
  bool loadRLCfg(ros::NodeHandle& nh) override;
  void computeActions() override;
  void computeObservation() override;
  void computeDepthLatent();

  void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg);
  std::vector<float> cropDepthImage(const std::vector<float>& image, int width, int height, int left, int right, int top, int bottom);

 private:
  std::string policyFilePath_;
  std::shared_ptr<Ort::Env> onnxEnvPtr_;
  std::unique_ptr<Ort::Session> policySessionPtr_;
  std::vector<const char*> policyInputNames_;
  std::vector<const char*> policyOutputNames_;
  std::vector<std::vector<int64_t>> policyInputShapes_;
  std::vector<std::vector<int64_t>> policyOutputShapes_;

  std::string depthEncoderPath_;
  std::shared_ptr<Ort::Env> depthEncoderOnnxEnvPrt_;
  std::unique_ptr<Ort::Session> depthEncoderSessionPtr_;
  std::vector<const char*> depthEncoderInputNames_;
  std::vector<const char*> depthEncoderOutputNames_;
  std::vector<std::vector<int64_t>> depthEncoderInputShapes_;
  std::vector<std::vector<int64_t>> depthEncoderOutputShapes_;

  int actionsSize_;
  int depthLatentSize_;
  int observationsSize_;
  int proprioObservationSize_;
  int combinedImageProPrioSize_;
  int proprioObsHistoryLen_;
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> observations_;
  std::vector<tensor_element_t> depthLatent_;
  std::vector<tensor_element_t> combinedImageProprio_;
  std::vector<tensor_element_t> combinedObsDepthLatent_;
  vector_t proprioObsHistoryBuffer_;

  bool isFirstRecObs_{true};
  bool isFirstRecDepth_{true};

  int numPixels_;
  float farClip_;
  float nearClip_;
  int depthBufferLen_;
  std::vector<int> depthOriginalShape_;
  std::vector<int> depthResizedShape_;
  std::shared_ptr<std::deque<std::vector<float>>> depthBufferPtr_;

  ros::Subscriber depthImageSub_;
  ros::Publisher resizedDepthImagePub_;
};

}  // namespace legged