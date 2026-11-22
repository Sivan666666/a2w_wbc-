from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_Q = np.array([0., .8, -1.5] * 4 + [0.] * 10, dtype=np.float32)
KP = np.array([120.] * 12 + [50., 50., 80., 30., 30., 20.] + [0.] * 4)
KD = np.array([3.] * 12 + [3., 2., 3., 3., 2.5, 1.] + [.6] * 4)
JOINTS = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
          for joint in ("hip", "thigh", "calf")] + [f"joint{i}" for i in range(1, 7)] + [
          f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")]


class Observation:
    """IsaacLab history is per term, oldest to newest; 78 x 3 = 234."""
    def __init__(self):
        self.history = None
        self.previous_action = np.zeros(22, dtype=np.float32)

    def build(self, q, dq, quaternion_wxyz, gyro, velocity, ee):
        w, x, y, z = quaternion_wxyz
        gravity = Rotation.from_quat([x, y, z, w]).inv().apply([0., 0., -1.])
        terms = [np.asarray(gyro) * .2, gravity, np.asarray(q)[:18] - DEFAULT_Q[:18],
                 np.asarray(dq) * .05, self.previous_action.copy(), velocity, ee]
        terms = [np.asarray(t, dtype=np.float32).copy() for t in terms]
        if self.history is None:
            self.history = [deque([t.copy() for _ in range(3)], maxlen=3) for t in terms]
        else:
            for history, term in zip(self.history, terms):
                history.append(term)
        return np.concatenate([np.concatenate(h) for h in self.history])


def action_targets(action):
    """Exact training action processing; no additional runtime action clipping."""
    q = DEFAULT_Q.copy()
    dq = np.zeros(22, dtype=np.float32)
    q[:18] = np.clip(DEFAULT_Q[:18] + .25 * action[:18], -10., 10.)
    dq[18:] = np.clip(5. * action[18:], -100., 100.)
    return q, dq


class Actor:
    def __init__(self, path):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.model = torch.jit.load(str(path), map_location="cpu").eval()

    def __call__(self, observation):
        with self.torch.inference_mode():
            return self.model(self.torch.from_numpy(observation)[None]).numpy()[0]
