"""Offline inference helpers; no hardware, transport, or motor-control dependency.

Inputs are the same scaled, ordered observation terms as the Isaac adapter.
History uses Isaac's term-major/oldest-to-newest layout, NOT frame-major.
"""
import torch
from .core import impedance_external_command

POLICY_TERMS = (
    ('base_ang_vel', 3), ('projected_gravity', 3), ('joint_pos', 18),
    ('joint_vel', 22), ('actions', 22), ('velocity_commands', 3),
    ('pos_commands', 7), ('ee_force', 4),
)


class PolicyHistory:
    def __init__(self, num_envs, device='cpu', length=10):
        self.length = length
        self.buffer = torch.zeros(num_envs, length, sum(d for _, d in POLICY_TERMS), device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def push(self, frame, reset=None):
        if frame.shape != self.buffer[:, 0].shape:
            raise ValueError('Expected one scaled 82-dimensional observation frame per environment')
        first = ~self.initialized
        if reset is not None:
            first = first | reset.to(dtype=torch.bool, device=frame.device)
        self.buffer = torch.cat((self.buffer[:, 1:], frame[:, None]), dim=1)
        self.buffer[first] = frame[first, None].expand(-1, self.length, -1)
        self.initialized[:] = True
        return torch.cat([x.flatten(1) for x in self.buffer.split([d for _, d in POLICY_TERMS], dim=-1)], dim=-1)

    @staticmethod
    def latest_frame(history, length=10):
        return torch.cat([x.reshape(len(history), length, d)[:, -1]
                          for x, (_, d) in zip(history.split([length*d for _, d in POLICY_TERMS], dim=-1), POLICY_TERMS)], dim=-1)


class ImpedanceCommand:
    """Force command from estimated EE position in a fixed yaw-aligned local frame.

    Use a fixed reference frame across calls (or transform estimates first).
    Raw differentiation of positions in a rotating/moving frame is not world EE
    velocity. This helper intentionally does not consume simulator ground truth.
    """
    def __init__(self, dt=.02, kp=100., kd=10., limit=70.):
        if dt <= 0 or kp < 0 or kd < 0 or limit <= 0:
            raise ValueError('Invalid impedance parameters')
        self.dt, self.kp, self.kd, self.limit = dt, kp, kd, limit
        self.previous = None

    def __call__(self, estimated_position, target, reset=None):
        velocity = torch.zeros_like(estimated_position) if self.previous is None else (estimated_position-self.previous)/self.dt
        if reset is not None:
            velocity[reset] = 0
        self.previous = estimated_position.detach().clone()
        return impedance_external_command(estimated_position, velocity, target, self.kp, self.kd, self.limit)
