"""Semantic actions for the single Franka Panda in LIBERO.

EEF translations, rotations, and camera actions all use robot base axes:
FWD/RIGHT/UP are base +X/+Y/+Z, independent of wrist and camera orientation.
Metric step sizes and optional rotation/camera actions are set in context.py.
"""

# Incremental end-effector translations.
MOVE_ATOMS = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
)

# End-effector rotations about base-fixed axes through the TCP.
RT_ATOMS = (
    "RT_ROLL_LEFT",
    "RT_ROLL_RIGHT",
    "RT_PITCH_FWD",
    "RT_PITCH_BACK",
    "RT_YAW_CW",
    "RT_YAW_CCW",
)

# External-camera active perception, in the robot base frame.
CAMERA_ATOMS = (
    "CAM_UP",
    "CAM_DOWN",
    "CAM_FWD",
    "CAM_BACK",
    "CAM_LEFT",
    "CAM_RIGHT",
    "CAM_YAW_CW",
    "CAM_YAW_CCW",
)

# Model decisions; STOP is reserved for recovery while a grasp settles.
GRIPPER_ATOMS = ("GRASP", "RELEASE")
DONE_ATOM = "DONE"
STOP_ATOM = "STOP"
CONTROLLER_ATOMS = MOVE_ATOMS + GRIPPER_ATOMS + (DONE_ATOM,)
