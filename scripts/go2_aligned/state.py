"""Robot-specific layout; algorithm source uses these dimensions."""
STATE_DIM = 86
NUM_JOINTS = 22
CONTEXT_DIM = 766
POSE_DELTA_DIM = 3
Z = slice(0, 1)
GRAV = slice(1, 4)
V_YAW = slice(4, 7)
W_B = slice(7, 10)
Q = slice(10, 28)
QD = slice(28, 50)
HIPS = slice(50, 62)
KNEES = slice(62, 74)
FEET = slice(74, 86)
