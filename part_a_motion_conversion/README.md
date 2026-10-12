# Part A — Human Motion to Robot

This project uses **UltimateBots Studio** to generate a humanoid motion, then plays it on the Unitree G1 model in **MuJoCo** and renders the result as a video.

## Files

| File             | Description                                                          |
| ---------------- | -------------------------------------------------------------------- |
| `motion.csv`     | Generated motion in LAFAN CSV format                                 |
| `motion.mp4`     | Rendered robot motion                                                |
| `prompt.txt`     | Prompt used to generate the motion                                   |
| `play_motion.py` | Script that loads the robot, plays the motion, and renders the video |

## Motion

- **Robot:** Unitree G1
- **Duration:** 8 seconds
- **Frames:** 240
- **Frame rate:** 30 FPS

The robot model is loaded from the `resources/` directory in the parent `unitree_rl_gym` repository.

## How to Run

Activate the Python virtual environment and install the required dependencies.

From the `part_a_motion_conversion` directory, run:

```powershell
py play_motion.py
```

The script loads `motion.csv`, applies each frame to the robot's configuration, and saves the rendered video as `motion.mp4`.

## How I Identified the CSV Columns

I inspected the CSV dimensions and compared them with the Unitree G1 model's configuration in MuJoCo.

The CSV contains 240 frames with 36 values per frame. The model has 36 configuration coordinates (`nq = 36`), 35 velocity coordinates (`nv = 35`), and 29 actuators (`nu = 29`).

The model's joint addresses show that seven configuration values belong to the floating base: three for position and four for orientation. The remaining 29 values correspond to the robot's articulated joints.

Based on this structure, the playback script interprets the CSV columns as follows:

| Columns | Interpretation                                |
| ------- | --------------------------------------------- |
| `0–2`   | Root position (x, y, z)                       |
| `3–6`   | Root orientation quaternion                   |
| `7–35`  | Joint positions for the 29 articulated joints |

I checked the quaternion norms across all frames and compared the joint values with the model's configured joint limits. The quaternion norms remained approximately 1, and no joint-limit violations were detected.
