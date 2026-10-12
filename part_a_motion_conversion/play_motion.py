from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

ROBOT_XML = (
    ROOT
    / "resources"
    / "robots"
    / "g1_description"
    / "g1_29dof.xml"
)

CSV_FILE = SCRIPT_DIR / "motion.csv"
OUTPUT_FILE = SCRIPT_DIR / "motion.mp4"

FPS = 30
WIDTH = 640
HEIGHT = 480

# CSV quaternion order:
# "xyzw" means x, y, z, w.
# "wxyz" means w, x, y, z.
CSV_QUATERNION_ORDER = "xyzw"

# Camera settings
CAMERA_LOOKAT = [0.0, 0.0, 0.8]
CAMERA_DISTANCE = 2.5
CAMERA_AZIMUTH = 135
CAMERA_ELEVATION = -15


def load_motion(csv_path: Path) -> np.ndarray:
    """Load and validate the motion CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Motion CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] != 36:
        raise ValueError(
            f"Expected 36 columns in the motion CSV, "
            f"but found {df.shape[1]}."
        )

    if df.empty:
        raise ValueError("The motion CSV contains no frames.")

    motion = df.to_numpy(dtype=np.float64)

    if not np.isfinite(motion).all():
        raise ValueError("The motion CSV contains NaN or infinite values.")

    print(f"Motion CSV: {csv_path}")
    print(f"Motion frames: {len(motion)}")
    print(f"CSV shape: {motion.shape}")
    print(f"Expected video duration: {len(motion) / FPS:.2f} seconds")

    return motion


def configure_camera() -> mujoco.MjvCamera:
    """Create a fixed camera for the recording."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    camera.lookat[:] = CAMERA_LOOKAT
    camera.distance = CAMERA_DISTANCE
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION

    return camera


def get_joint_mapping(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """
    Map CSV columns 7–35 to the model's 29 non-free joints.

    This assumes the CSV joint order matches the order of non-free
    joints in the MuJoCo model.
    """
    free_joint_ids = [
        jid
        for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
    ]

    if not free_joint_ids:
        raise RuntimeError("No floating-base free joint was found.")

    if len(free_joint_ids) != 1:
        raise RuntimeError(
            f"Expected one free joint, found {len(free_joint_ids)}."
        )

    joint_ids = [
        jid
        for jid in range(model.njnt)
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE
    ]

    if len(joint_ids) != 29:
        raise RuntimeError(
            f"Expected 29 non-free joints, found {len(joint_ids)}."
        )

    mapping = []

    for column, jid in zip(range(7, 36), joint_ids):
        joint_type = model.jnt_type[jid]

        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise RuntimeError(
                f"Joint {jid} is not a single-DoF hinge or slide joint."
            )

        qpos_address = int(model.jnt_qposadr[jid])
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, jid
        )

        mapping.append((column, qpos_address))
        print(f"CSV column {column} -> {joint_name}")

    return mapping


def set_robot_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    row: np.ndarray,
    root_qpos_address: int,
    joint_mapping: list[tuple[int, int]],
    frame_index: int,
) -> None:
    """Apply one CSV frame to the robot's MuJoCo configuration."""

    # Root position: CSV columns 0, 1, 2.
    data.qpos[root_qpos_address:root_qpos_address + 3] = row[:3]

    # Root quaternion: MuJoCo requires w, x, y, z.
    quaternion = row[3:7].copy()

    if CSV_QUATERNION_ORDER == "xyzw":
        quaternion = quaternion[[3, 0, 1, 2]]
    elif CSV_QUATERNION_ORDER != "wxyz":
        raise ValueError(
            "CSV_QUATERNION_ORDER must be 'xyzw' or 'wxyz'."
        )

    quaternion_norm = np.linalg.norm(quaternion)

    if quaternion_norm < 1e-8:
        raise ValueError(
            f"Invalid root quaternion at frame {frame_index}."
        )

    data.qpos[
        root_qpos_address + 3:root_qpos_address + 7
    ] = quaternion / quaternion_norm

    # Joint positions: CSV columns 7–35.
    for column, qpos_address in joint_mapping:
        data.qpos[qpos_address] = row[column]

    # Update derived positions, orientations, and geometry.
    mujoco.mj_forward(model, data)


def main() -> None:
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(f"Robot XML not found: {ROBOT_XML}")

    motion = load_motion(CSV_FILE)

    print(f"\nLoading robot model: {ROBOT_XML.name}")

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)

    print(f"Model joints: {model.njnt}")
    print(f"Position coordinates (nq): {model.nq}")
    print(f"Velocity coordinates (nv): {model.nv}")
    print(f"Actuators: {model.nu}")

    if model.nq != 36:
        raise ValueError(
            f"Expected nq=36 for this CSV layout, found nq={model.nq}."
        )

    free_joint_ids = [
        jid
        for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
    ]

    if len(free_joint_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one free joint, found {len(free_joint_ids)}."
        )

    root_qpos_address = int(
        model.jnt_qposadr[free_joint_ids[0]]
    )

    joint_mapping = get_joint_mapping(model)
    camera = configure_camera()

    renderer = None
    writer = None

    try:
        renderer = mujoco.Renderer(
            model,
            height=HEIGHT,
            width=WIDTH,
        )

        # H.264 MP4 for compatibility with common media players
        # and web previews.
        writer = imageio.get_writer(
            str(OUTPUT_FILE),
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=16,
        )

        print(f"\nRendering {len(motion)} frames...")
        print(f"Output: {OUTPUT_FILE}")

        for frame_index, row in enumerate(motion):
            set_robot_pose(
                model=model,
                data=data,
                row=row,
                root_qpos_address=root_qpos_address,
                joint_mapping=joint_mapping,
                frame_index=frame_index,
            )

            renderer.update_scene(data, camera=camera)

            # MuJoCo returns RGB frames; ImageIO expects RGB.
            rgb = renderer.render()
            writer.append_data(rgb)

            if (frame_index + 1) % 25 == 0:
                print(
                    f"Rendered {frame_index + 1}/{len(motion)} frames"
                )

    finally:
        if writer is not None:
            writer.close()

        if renderer is not None:
            renderer.close()

    if not OUTPUT_FILE.is_file() or OUTPUT_FILE.stat().st_size == 0:
        raise RuntimeError(
            f"Video output was not created correctly: {OUTPUT_FILE}"
        )

    print("\nVideo rendering completed.")
    print(f"Video saved to: {OUTPUT_FILE}")
    print(f"Video size: {OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Video frames: {len(motion)}")
    print(f"Video duration: {len(motion) / FPS:.2f} seconds")
    print(f"Video FPS: {FPS}")


if __name__ == "__main__":
    main()
