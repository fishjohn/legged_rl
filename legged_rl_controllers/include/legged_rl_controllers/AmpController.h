//
// Created by luohx on 23-8-29.
//

#pragma once

#include "legged_rl_controllers/RLControllerBase.h"

namespace legged {
using namespace ocs2;
using namespace legged_robot;

class AmpController : public RLControllerBase {
  using tensor_element_t = float;

 public:
  AmpController() = default;
  ~AmpController() override = default;

 protected:
  bool loadModel(ros::NodeHandle& nh) override;
  bool loadRLCfg(ros::NodeHandle& nh) override;
  void computeActions() override;
  void computeObservation() override;
  void handleWalkMode() override;

 private:
  // onnx policy model
  std::string policyFilePath_;
  std::shared_ptr<Ort::Env> onnxEnvPrt_;
  std::unique_ptr<Ort::Session> sessionPtr_;
  std::vector<const char*> inputNames_;
  std::vector<const char*> outputNames_;
  std::vector<std::vector<int64_t>> inputShapes_;
  std::vector<std::vector<int64_t>> outputShapes_;

  vector3_t baseLinVel_;
  vector3_t basePosition_;
  vector_t lastActions_;
  vector_t defaultJointAngles_;

  int actionsSize_;
  int observationSize_;
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> observations_;
};

}  // namespace legged