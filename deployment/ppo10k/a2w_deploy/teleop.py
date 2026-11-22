import json
import os
import select
import socket
import sys
import termios
import tty
import numpy as np
from scipy.spatial.transform import Rotation


class Command:
    def __init__(self, config):
        self.config = config
        self.velocity = np.zeros(3)
        self.ee = np.array(config["ee_home"], dtype=float)
        self.home_ee = self.ee.copy()
        self.gripper = config["gripper_closed"]
        self.clutch = None

    def home(self):
        self.velocity[:] = 0
        self.ee[:] = self.home_ee
        self.clutch = None

    def capture_home(self, ee):
        self.home_ee[:] = ee
        self.home()

    def key(self, key):
        cfg = self.config
        translations = {"i": (0, 1), "k": (0, -1), "j": (1, 1), "l": (1, -1),
                        "u": (2, 1), "o": (2, -1)}
        rotations = {"1": (0, 1), "2": (0, -1), "3": (1, 1), "4": (1, -1),
                     "5": (2, 1), "6": (2, -1)}
        base = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1),
                "q": (2, 1), "e": (2, -1)}
        if key in translations:
            axis, sign = translations[key]
            self.ee[axis] += sign * cfg["position_step"]
        elif key in rotations:
            axis, sign = rotations[key]
            delta = np.zeros(3)
            delta[axis] = sign * cfg["rotation_step"]
            current = Rotation.from_quat(self.ee[[4, 5, 6, 3]])
            self.set_rotation(Rotation.from_rotvec(delta) * current)
        elif key in base:
            axis, sign = base[key]
            self.velocity[axis] += sign * cfg["velocity_step"][axis]
        elif key == " ":
            self.velocity[:] = 0
        elif key == "h":
            self.home()
        elif key == "n":
            self.gripper = cfg["gripper_open"]
        elif key == "m":
            self.gripper = cfg["gripper_closed"]

    def set_rotation(self, rotation):
        xyzw = rotation.as_quat()
        # Training uses quat_unique (w >= 0).
        if xyzw[3] < 0:
            xyzw = -xyzw
        self.ee[3:] = xyzw[[3, 0, 1, 2]]

    def vr(self, controllers):
        """PXREA raw Controller JSON. Pose: x,y,z,qx,qy,qz,qw.

        Position deltas become command dx_link0, dy_link0, dz_world.
        Rotation deltas become root-relative desired EE rotation, as in reward.
        """
        right = controllers["right"]
        left = controllers.get("left", {})
        cfg = self.config
        self.velocity[:] = [left.get("axisY", 0.) * cfg["vr_vx"],
                            0., -left.get("axisX", 0.) * cfg["vr_yaw"]]
        trigger = right.get("trigger", 0.)
        self.gripper = cfg["gripper_closed"] + trigger * (
            cfg["gripper_open"] - cfg["gripper_closed"])
        if right.get("secondaryButton", False):
            self.home()
            return
        if right.get("grip", 0.) <= .5:
            self.clutch = None
            return
        pose = np.array([float(v) for v in right["pose"].split(",")])
        rotation = Rotation.from_quat(pose[3:7]).as_matrix()
        if self.clutch is None:
            self.clutch = (pose[:3], rotation, self.ee.copy())
        p0, r0, ee0 = self.clutch
        # Default OpenXR axes: right/+X, up/+Y, backward/+Z -> robot FLU.
        basis = np.array(cfg["vr_basis"])
        self.ee[:3] = ee0[:3] + cfg["vr_scale"] * basis @ (pose[:3] - p0)
        delta = basis @ (rotation @ r0.T) @ basis.T
        self.set_rotation(Rotation.from_matrix(delta) * Rotation.from_quat(ee0[[4, 5, 6, 3]]))


class Keyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self):
        return os.read(self.fd, 1).decode() if select.select([self.fd], [], [], 0)[0] else None

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class VRReceiver:
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.setblocking(False)

    def drain(self):
        # Consume each packet so grip release/re-engage edges are not lost.
        while select.select([self.socket], [], [], 0)[0]:
            packet, _ = self.socket.recvfrom(65535)
            yield json.loads(packet)

    def close(self):
        self.socket.close()
