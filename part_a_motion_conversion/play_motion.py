
import cv2
import mujoco
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
ROBOT_XML = ROOT / "resources" / "robots" / "g1_description" / "g1_29dof.xml"
CSV_FILE = Path(__file__).resolve().parent / "motion.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "motion.mp4"

FPS = 30
WIDTH = 640
HEIGHT = 480

# Change to "wxyz" if the CSV quaternion is stored as w, x, y, z.
CSV_QUATERNION_ORDER = "xyzw"


def main():
    # --------------------------------------------------
    # Load robot model
    # --------------------------------------------------

    if not ROBOT_XML.exists():
        raise FileNotFoundError(f"Robot XML not found: {ROBOT_XML}")

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)

    print(f"Robot model: {ROBOT_XML.name}")
    print(f"Model joints: {model.njnt}")
    print(f"Position coordinates (nq): {model.nq}")
    print(f"Actuators: {model.nu}")

    # --------------------------------------------------
    # Load CSV without headers
    # --------------------------------------------------

    df = pd.read_csv(CSV_FILE, header=None)

    if df.shape[1] != 36:
        raise ValueError(
            f"Expected 36 CSV columns, got {df.shape[1]}"
        )

    df = df.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("CSV contains NaN or infinite values.")

    print(f"Motion frames: {len(df)}")
    print(f"CSV shape: {df.shape}")

    # --------------------------------------------------
    # Find floating-base root joint
    # --------------------------------------------------

    free_joint_id = None

    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            free_joint_id = jid
            break

    if free_joint_id is None:
        raise RuntimeError("No free joint found in the robot model.")

    root_adr = model.jnt_qposadr[free_joint_id]

    # --------------------------------------------------
    # Identify non-free joints
    # --------------------------------------------------

    joint_ids = [
        jid
        for jid in range(model.njnt)
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE
    ]

    if len(joint_ids) != 29:
        raise RuntimeError(
            f"Expected 29 non-free joints, found {len(joint_ids)}."
        )

    joint_columns = list(range(7, 36))

    print("\nJoint mapping assumption:")
    for column, jid in zip(joint_columns, joint_ids):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, jid
        )
        print(f"  CSV column {column} -> {name}")

    # --------------------------------------------------
    # Set up renderer and video writer
    # --------------------------------------------------

    renderer = mujoco.Renderer(
        model,
        height=HEIGHT,
        width=WIDTH,
    )

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 2.5
    camera.azimuth = 135
    camera.elevation = -15
    camera.lookat[:] = [0, 0, 0.8]

    writer = cv2.VideoWriter(
        str(OUTPUT_FILE),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )

    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Could not create video: {OUTPUT_FILE}")

    # --------------------------------------------------
    # Render every motion frame
    # --------------------------------------------------

    try:
        for frame_index in range(len(df)):
            row = df.iloc[frame_index].to_numpy(dtype=float)

            # Root position: columns 0, 1, 2
            data.qpos[root_adr:root_adr + 3] = row[0:3]

            # Root quaternion: columns 3, 4, 5, 6
            quat = row[3:7].copy()

            if CSV_QUATERNION_ORDER == "xyzw":
                quat = quat[[3, 0, 1, 2]]
            elif CSV_QUATERNION_ORDER != "wxyz":
                raise ValueError("Invalid quaternion order setting.")

            quat_norm = np.linalg.norm(quat)

            if quat_norm < 1e-8:
                raise ValueError(
                    f"Invalid quaternion at frame {frame_index}"
                )

            data.qpos[root_adr + 3:root_adr + 7] = (
                quat / quat_norm
            )

            # Joint positions: columns 7 through 35
            for column, jid in zip(joint_columns, joint_ids):
                joint_type = model.jnt_type[jid]

                if joint_type not in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ):
                    raise RuntimeError(
                        f"Unsupported joint type for joint ID {jid}"
                    )

                qpos_adr = model.jnt_qposadr[jid]
                data.qpos[qpos_adr] = row[column]

            # Update robot geometry
            mujoco.mj_forward(model, data)

            # Render frame
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

            if (frame_index + 1) % 25 == 0:
                print(
                    f"Rendered {frame_index + 1}/{len(df)} frames"
                )

    finally:
        writer.release()
        renderer.close()

    print(f"\nVideo saved to: {OUTPUT_FILE}")
    print(f"Duration: {len(df) / FPS:.2f} seconds")


if __name__ == "__main__":
    main()
