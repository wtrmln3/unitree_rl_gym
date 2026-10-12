import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion

    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ], dtype=np.float32)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    root = Path(LEGGED_GYM_ROOT_DIR)
    config_path = (
        root / "deploy" / "deploy_mujoco" / "configs" / "g1_circle.yaml"
    )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    policy_path = config["policy_path"].replace(
        "{LEGGED_GYM_ROOT_DIR}", str(root)
    )
    xml_path = config["xml_path"].replace(
        "{LEGGED_GYM_ROOT_DIR}", str(root)
    )

    simulation_duration = float(config["simulation_duration"])
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = float(config["ang_vel_scale"])
    dof_pos_scale = float(config["dof_pos_scale"])
    dof_vel_scale = float(config["dof_vel_scale"])
    action_scale = float(config["action_scale"])
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    if model.nu != len(default_angles):
        raise ValueError(
            f"Model has {model.nu} actuators, but default_angles has "
            f"{len(default_angles)} values."
        )

    if len(kps) != model.nu or len(kds) != model.nu:
        raise ValueError(
            "The lengths of kps and kds must match the number of actuators."
        )

    if num_obs != 47 or num_actions != model.nu:
        raise ValueError(
            f"Unexpected policy dimensions: num_obs={num_obs}, "
            f"num_actions={num_actions}, actuators={model.nu}."
        )

    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    output_path = Path(__file__).resolve().parent / "circle_walking.mp4"
    video_fps = 25
    frame_interval = round(1 / (simulation_dt * video_fps))

    frames = []
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    renderer = mujoco.Renderer(model, height=480, width=640)

    # Fixed three-quarter camera view for the recording.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.8]
    camera.distance = 3.0
    camera.azimuth = 135
    camera.elevation = -15

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print(f"Target simulation duration: {simulation_duration:.1f} s")
            print("Recording started. Keep the viewer open until it finishes.")

            while viewer.is_running() and data.time < simulation_duration:
                step_start = time.perf_counter()

                # Apply joint torques.
                tau = pd_control(
                    target_dof_pos,
                    data.qpos[7:],
                    kps,
                    np.zeros_like(kds),
                    data.qvel[6:],
                    kds,
                )
                data.ctrl[:] = tau
                mujoco.mj_step(model, data)
                counter += 1

                # Update the walking policy at its control frequency.
                if counter % control_decimation == 0:
                    qj = data.qpos[7:].copy()
                    dqj = data.qvel[6:].copy()
                    quat = data.qpos[3:7]
                    omega = data.qvel[3:6].copy()

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega *= ang_vel_scale

                    period = 0.8
                    phase = ((counter * simulation_dt) % period) / period

                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9:21] = qj
                    obs[21:33] = dqj
                    obs[33:45] = action
                    obs[45] = np.sin(2 * np.pi * phase)
                    obs[46] = np.cos(2 * np.pi * phase)

                    obs_tensor = torch.from_numpy(obs.copy()).unsqueeze(0)

                    with torch.no_grad():
                        action = (
                            policy(obs_tensor)
                            .cpu()
                            .numpy()
                            .reshape(-1)
                            .astype(np.float32)
                        )

                    target_dof_pos = (
                        action * action_scale + default_angles
                    )

                # Capture one frame per 1/video_fps simulated seconds.
                if counter % frame_interval == 0:
                    renderer.update_scene(data, camera=camera)
                    frames.append(renderer.render())

                viewer.sync()

                # Pace the simulation close to real time.
                remaining = simulation_dt - (
                    time.perf_counter() - step_start
                )
                if remaining > 0:
                    time.sleep(remaining)

                if counter % 5000 == 0:
                    print(
                        f"Simulation: {data.time:.1f} / "
                        f"{simulation_duration:.1f} seconds | "
                        f"Frames: {len(frames)}"
                    )

        print(f"Simulation ended at: {data.time:.2f} seconds")

        if not frames:
            raise RuntimeError(
                "No video frames were recorded. Check the simulation and viewer."
            )

        imageio.mimsave(
            str(output_path),
            frames,
            fps=video_fps,
            macro_block_size=16,
        )

        print(f"Video saved to: {output_path}")
        print(f"Recorded frames: {len(frames)}")
        print(f"Video duration: {len(frames) / video_fps:.2f} seconds")

    finally:
        renderer.close()


if __name__ == "__main__":
    main()
