//
// Created by luohx on 22-12-5.
//

#pragma once

#include <controller_interface/multi_interface_controller.h>
#include <gazebo_msgs/ModelStates.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <legged_common/hardware_interface/ContactSensorInterface.h>
#include <legged_common/hardware_interface/HybridJointInterface.h>
#include <legged_estimation/LinearKalmanFilter.h>
#include <legged_estimation/StateEstimateBase.h>
#include <legged_interface/LeggedInterface.h>
#include <std_msgs/Float32MultiArray.h>

#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <Eigen/Geometry>

namespace legged {
using namespace ocs2;
using namespace legged_robot;

struct RobotCfg {
  struct ControlCfg {
    float stiffness;
    float damping;
    float actionScale;
    int decimation;
  };

  struct InitState {
    // default joint angles
    scalar_t LF_HAA_joint;
    scalar_t LF_HFE_joint;
    scalar_t LF_KFE_joint;

    scalar_t LH_HAA_joint;
    scalar_t LH_HFE_joint;
    scalar_t LH_KFE_joint;

    scalar_t RF_HAA_joint;
    scalar_t RF_HFE_joint;
    scalar_t RF_KFE_joint;

    scalar_t RH_HAA_joint;
    scalar_t RH_HFE_joint;
    scalar_t RH_KFE_joint;
  };

  struct ObsScales {
    scalar_t linVel;
    scalar_t angVel;
    scalar_t dofPos;
    scalar_t dofVel;
    scalar_t heightMeasurements;
  };

  scalar_t clipActions;
  scalar_t clipObs;

  InitState initState;
  ObsScales obsScales;
  ControlCfg controlCfg;
};

class LeggedRLController
    : public controller_interface::MultiInterfaceController<HybridJointInterface, hardware_interface::ImuSensorInterface,
                                                            ContactSensorInterface> {
  using tensor_element_t = float;
  using ObsScales = RobotCfg::ObsScales;
  using ControlCfg = RobotCfg::ControlCfg;
  using InitState = RobotCfg::InitState;

  enum class Mode : uint8_t { LIE, STAND, WALK };

 public:
  LeggedRLController() = default;
  ~LeggedRLController() override;
  bool init(hardware_interface::RobotHW* robotHw, ros::NodeHandle& controllerNH) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void starting(const ros::Time& time) override;
  void stopping(const ros::Time& time) override;

 protected:
  virtual void updateStateEstimation(const ros::Time& time, const ros::Duration& period);

  virtual void setupLeggedInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile,
                                    bool verbose);
  virtual void setupStateEstimate(const std::string& taskFile, bool verbose);

  bool parseCfg(ros::NodeHandle& nh);
  void loadPolicyModel(const std::string& policy_file_path);
  void computeActions();
  void computeObservation(const ros::Time& time, const ros::Duration& period);

  void cmdVelCallback(const geometry_msgs::Twist& msg);
  void baseStateRecCallback(const gazebo_msgs::ModelStates& msg);

  std::shared_ptr<StateEstimateBase> stateEstimate_;
  std::shared_ptr<LeggedInterface> leggedInterface_;
  std::shared_ptr<PinocchioEndEffectorKinematics> eeKinematicsPtr_;

  // state estimation
  SystemObservation currentObservation_;
  vector_t measuredRbdState_;
  std::shared_ptr<CentroidalModelRbdConversions> rbdConversions_;

  std::vector<scalar_t> initJointAngles_;
  std::vector<HybridJointHandle> hybridJointHandles_;
  hardware_interface::ImuSensorHandle imuSensorHandles_;
  std::vector<ContactSensorHandle> contactHandles_;

 private:
  std::atomic_bool controllerRunning_;
  int loopCount_;
  Mode mode_;

  // publisher & subscriber
  ros::Subscriber cmdVelSub_;
  ros::Subscriber baseStateSub_;

  // stand
  scalar_t standPercent_;
  scalar_t standDuration_;

  // onnx policy model
  std::string policyFilePath_;
  std::shared_ptr<Ort::Env> onnxEnvPrt_;
  std::unique_ptr<Ort::Session> sessionPtr_;
  std::vector<const char*> inputNames_;
  std::vector<const char*> outputNames_;
  std::vector<std::vector<int64_t>> inputShapes_;
  std::vector<std::vector<int64_t>> outputShapes_;

  int actionsSize_;
  int observationSize_;
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> observations_;

  // temp state
  RobotCfg robotCfg_{};
  vector_t rbdState_;
  vector3_t command_;
  vector3_t baseLinVel_;
  vector3_t basePosition_;
  vector_t lastActions_;
  vector_t defaultJointAngles_;
};

}  // namespace legged
