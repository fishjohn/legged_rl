//
// Created by luohx on 23-11-7.
//

#pragma once

#include <sensor_msgs/Image.h>

#include "legged_rl_controllers/RLControllerBase.h"

namespace legged {
using namespace ocs2;
using namespace legged_robot;
using tensor_element_t = float;

class ParkourController : public RLControllerBase {
 public:
  ParkourController() = default;
  ~ParkourController() override = default;

 protected:
  bool loadModel(ros::NodeHandle& nh) override;
  bool loadRLCfg(ros::NodeHandle& nh) override;
  void computeActions() override;
  void computeObservation() override;
  void handleWalkMode() override;
  void computeDepthLatent();

  std::vector<float> cropDepthImage(const std::vector<float>& image, int width, int height, int left, int right, int top, int bottom);
  void resizeTransformDepthImage();
  void normalizeDepthImage();
  void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg);

  vector_t reindexFeetOrder(const vector_t& vec);
  vector_t reindexJointOrder(const vector_t& vec);
  auto reindexActions(const std::vector<tensor_element_t>& actions) -> std::vector<tensor_element_t>;

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

  vector3_t baseLinVel_;
  vector3_t basePosition_;
  vector_t lastActions_;
  vector_t defaultJointAngles_;

  int actionsSize_;
  int depthLatentSize_;
  int predictedYawSize_;
  int observationsSize_;
  int proprioObservationSize_;
  int combinedImageProPrioSize_;
  int proprioObsHistoryLen_;
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> observations_;
  std::vector<tensor_element_t> depthLatent_;
  std::vector<tensor_element_t> predictedYaw_;
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
