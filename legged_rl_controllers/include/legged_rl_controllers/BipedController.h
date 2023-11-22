//
// Created by luohx on 23-11-15.
//

#pragma once

#include <controller_interface/multi_interface_controller.h>
#include <gazebo_msgs/ModelStates.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <legged_common/hardware_interface/ContactSensorInterface.h>
#include <legged_common/hardware_interface/HybridJointInterface.h>
#include <nav_msgs/Odometry.h>

#include <ocs2_core/Types.h>
#include <ocs2_legged_robot/common/Types.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <Eigen/Geometry>

namespace legged {
using namespace ocs2;
using namespace legged_robot;

struct RLRobotCfg {
  struct ControlCfg {
    float stiffness;
    float damping;
    float actionScale;
    int decimation;
    float user_torque_limit;
  };

  struct InitState {
    // default joint angles
    scalar_t L_HAA_joint;
    scalar_t L_HFE_joint;
    scalar_t L_KFE_joint;

    scalar_t R_HAA_joint;
    scalar_t R_HFE_joint;
    scalar_t R_KFE_joint;
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

class BipedController : public controller_interface::MultiInterfaceController<HybridJointInterface, hardware_interface::ImuSensorInterface,
                                                                              ContactSensorInterface> {
  using tensor_element_t = float;

 public:
  BipedController() = default;
  virtual ~BipedController() = default;
  virtual bool init(hardware_interface::RobotHW* robotHw, ros::NodeHandle& controllerNH);
  virtual void starting(const ros::Time& time);
  virtual void update(const ros::Time& time, const ros::Duration& period);

  virtual bool loadModel(ros::NodeHandle& nh);
  virtual bool loadRLCfg(ros::NodeHandle& nh);
  virtual void computeActions();
  virtual void computeObservation();

 protected:
  int64_t loopCount_;
  vector3_t command_;
  RLRobotCfg robotCfg_{};

  vector3_t baseLinVel_;
  vector3_t basePosition_;
  vector_t lastActions_;
  vector_t defaultJointAngles_;

  // hardware interface
  std::vector<HybridJointHandle> hybridJointHandles_;
  hardware_interface::ImuSensorHandle imuSensorHandles_;
  std::vector<ContactSensorHandle> contactHandles_;

  void cmdVelCallback(const geometry_msgs::Twist& msg);
  void stateUpdateCallback(const nav_msgs::Odometry& msg);

 private:
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
  double gait_index_;
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> observations_;

  ros::Subscriber gTStateSub_;
  ros::Subscriber cmdVelSub_;
};

template <typename T>
T square(T a) {
  return a * a;
}

template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 1> quatToZyx(const Eigen::Quaternion<SCALAR_T>& q) {
  Eigen::Matrix<SCALAR_T, 3, 1> zyx;

  SCALAR_T as = std::min(-2. * (q.x() * q.z() - q.w() * q.y()), .99999);
  zyx(0) = std::atan2(2 * (q.x() * q.y() + q.w() * q.z()), square(q.w()) + square(q.x()) - square(q.y()) - square(q.z()));
  zyx(1) = std::asin(as);
  zyx(2) = std::atan2(2 * (q.y() * q.z() + q.w() * q.x()), square(q.w()) - square(q.x()) - square(q.y()) + square(q.z()));
  return zyx;
}
}  // namespace legged
