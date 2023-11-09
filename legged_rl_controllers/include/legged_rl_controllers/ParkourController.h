//
// Created by luohx on 23-11-7.
//

#pragma once

#include "legged_rl_controllers/RLControllerBase.h"

namespace legged {
using namespace ocs2;
using namespace legged_robot;

class ParkourController : public RLControllerBase {
  using tensor_element_t = float;

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
  int observationsSize_;
  int proprioObservationSize_;
  int combinedImageProPrioSize_;
  std::vector<int> depthShape_;
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> observations_;
  std::vector<tensor_element_t> depthLatent_;
  std::vector<tensor_element_t> combinedImageProprio_;
};

}  // namespace legged
