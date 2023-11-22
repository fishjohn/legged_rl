//
// Created by luohx on 23-8-29.
//

#pragma once

#include "legged_rl_controllers/RLControllerBase.h"

namespace legged {
using namespace ocs2;
using namespace legged_robot;

class TrotController : public RLControllerBase {
  using tensor_element_t = float;

 public:
  TrotController() = default;
  ~TrotController() override = default;

 protected:
  bool loadModel(ros::NodeHandle& nh) override;
  bool loadRLCfg(ros::NodeHandle& nh) override;
  void computeActions() override;
  void computeEncoder();
  void computeObservation() override;
  void handleWalkMode() override;

 private:
  // onnx policy model
  std::shared_ptr<Ort::Env> onnxEnvPrt_;
  std::unique_ptr<Ort::Session> policySessionPtr_;
  std::unique_ptr<Ort::Session> encoderSessionPtr_;
  std::unique_ptr<Ort::Session> gaitGeneratorSessionPtr_;
  std::vector<const char*> policyInputNames_;
  std::vector<const char*> policyOutputNames_;
  std::vector<std::vector<int64_t>> policyInputShapes_;
  std::vector<std::vector<int64_t>> policyOutputShapes_;
  std::vector<const char*> encoderInputNames_;
  std::vector<const char*> encoderOutputNames_;
  std::vector<std::vector<int64_t>> encoderInputShapes_;
  std::vector<std::vector<int64_t>> encoderOutputShapes_;
  std::vector<const char*> gaitGeneratorInputNames_;
  std::vector<const char*> gaitGeneratorOutputNames_;
  std::vector<std::vector<int64_t>> gaitGeneratorInputShapes_;
  std::vector<std::vector<int64_t>> gaitGeneratorOutputShapes_;

  vector3_t baseLinVel_;
  vector3_t basePosition_;
  vector_t lastActions_;
  vector_t defaultJointAngles_;

  bool isfirstRecObs_{true};
  int actionsSize_;
  int observationSize_, commandsSize_;
  int obsHistoryLength_;
  int encoderIntputSize_, encoderOutputSize_, gaitGeneratorIntputSize_, gaitGeneratorOutputSize_;
  double gait_index_;
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> observations_, commands_;
  std::vector<tensor_element_t> encoderOut_, gaitGeneratorOut_;
  std::vector<tensor_element_t> proprioHistoryVector_;
  Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer_;
};

}  // namespace legged