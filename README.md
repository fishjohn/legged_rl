# legged_rl

---

## install

---

### ocs2

```
# Clone OCS2
git clone git@github.com:leggedrobotics/ocs2.git

# Clone pinocchio
git clone --recurse-submodules https://github.com/leggedrobotics/pinocchio.git

# Clone hpp-fcl
git clone --recurse-submodules https://github.com/leggedrobotics/hpp-fcl.git

# Clone ocs2_robotic_assets
git clone https://github.com/leggedrobotics/ocs2_robotic_assets.git

# Install dependencies
sudo apt install liburdfdom-dev liboctomap-dev libassimp-dev
```

### legged_control

```
# Clone legged_control
git clone git@github.com:qiayuanliao/legged_control.git
```

### legged_rl

```
# Clone legged_rl
git clone git@github.com:fishjohn/legged_rl.git
# Build legged_rl
catkin config -DCMAKE_BUILD_TYPE=RelWithDebInfo
catkin build legged_gazebo legged_unitree_description legged_rl_controllers legged_rl_description 
```

## Run

---

### Terminal 1

```
roslaunch legged_rl_description empty_world.launch
```

### Terminal 2

```
roslaunch legged_rl_controllers load_student_policy_controller.launch
```

```
rosservice call /controller_manager/switch_controller "start_controllers: ['controllers/student_policy_controller']                   
stop_controllers: ['']
strictness: 0
start_asap: false
timeout: 0.0" 
```

