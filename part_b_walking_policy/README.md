# Part B — Circular Walking

## What I changed

In `g1_circle.yaml`, I changed the walking command from `[0.5, 0, 0]` to `[0.5, 0, 0.8]`.

The three values represent forward velocity, lateral velocity, and yaw angular velocity. The nonzero yaw command tells the robot to turn while moving forward, producing a curved path. I reduced the forward command to make the circular movement easier to observe.

## 1. Policy inputs

The pretrained walking policy receives a 47-dimensional observation vector, assembled from the MuJoCo simulation:

- **3 values:** Angular velocity from the robot's simulated gyroscope state.
- **3 values:** Projected gravity direction, calculated from the robot's orientation quaternion.
- **3 values:** Desired velocity command, scaled using `cmd_scale` from the configuration.
- **12 values:** Current joint positions relative to the default joint angles.
- **12 values:** Current joint velocities.
- **12 values:** Previous policy actions.
- **2 values:** Sine and cosine of the gait phase.

These values describe the robot's current state, the requested movement, and its walking cycle.

## 2. Policy outputs and torque control

The policy outputs 12 actions, one for each controlled joint. The actions are scaled and added to the default joint angles to obtain target joint positions:

`target_dof_pos = action * action_scale + default_angles`

A proportional-derivative (PD) controller converts the target positions into torques:

`tau = kp * (target_q - q) + kd * (target_dq - dq)`

Here, `q` and `dq` are the current joint positions and velocities, while `kp` and `kd` are the proportional and derivative gains. The target joint velocities are set to zero. The resulting torques are sent to MuJoCo to simulate the robot's movement.

The policy generates joint targets; the PD controller calculates the torques needed to track them.
