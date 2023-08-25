//
// Created by luohx on 23-8-23.
//

#pragma once

#include "legged_rl_controllers/LeggedRLController.h"

namespace legged {
using namespace ocs2;
using namespace legged_robot;

class StudentPolicyController : public LeggedRLController {
 public:
  StudentPolicyController() = default;
  ~StudentPolicyController() override = default;
  bool init(hardware_interface::RobotHW* robotHw, ros::NodeHandle& controllerNH) override;
  void update(const ros::Time& time, const ros::Duration& period) override;

 protected:
  //  bool parseCfg(ros::NodeHandle& nh){} override;
  void computeActions() override;
  void computeObservation(const ros::Time& time, const ros::Duration& period) override;
  void loadModel(const std::string& policyFileDir);

  void cmdVelCallback(const geometry_msgs::Twist& msg) override;

 private:
  std::string policyModelPath_;
  std::string encoderModelPath_;
  std::shared_ptr<Ort::Env> onnxEnvPrt_;
  std::unique_ptr<Ort::Session> policySessionPtr_;
  std::unique_ptr<Ort::Session> encoderSessionPtr_;
  std::vector<const char*> policyInputNames_;
  std::vector<const char*> policyOutputNames_;
  std::vector<std::vector<int64_t>> policyInputShapes_;
  std::vector<std::vector<int64_t>> policyOutputShapes_;
  std::vector<const char*> encoderInputNames_;
  std::vector<const char*> encoderOutputNames_;
  std::vector<std::vector<int64_t>> encoderInputShapes_;
  std::vector<std::vector<int64_t>> encoderOutputShapes_;
  int64_t encoderBatchSize_, encoderSeqLength_, encoderInputDim_;

  bool isfirstRecObs_{true};
  Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer_;
};

}  // namespace legged