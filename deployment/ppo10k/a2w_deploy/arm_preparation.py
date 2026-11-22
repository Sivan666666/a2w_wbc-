"""Preparation trajectory only; PPO action processing remains in policy.py."""
import numpy as np


def quintic(q0, v0, q1, t, duration):
    """Match initial velocity, zero initial acceleration, and a stationary end."""
    q0, v0, q1 = (np.asarray(x) for x in (q0, v0, q1))
    if t >= duration:
        return q1.copy(), np.zeros_like(q1, dtype=float), np.zeros_like(q1, dtype=float)
    u = np.clip(t / duration, 0., 1.)
    delta, v = q1-q0, duration*v0
    a3, a4, a5 = 10*delta-6*v, -15*delta+8*v, 6*delta-3*v
    q = q0 + v*u + a3*u**3 + a4*u**4 + a5*u**5
    dq = (v + 3*a3*u**2 + 4*a4*u**3 + 5*a5*u**4)/duration
    ddq = (6*a3*u + 12*a4*u**2 + 20*a5*u**3)/duration**2
    return q, dq, ddq
