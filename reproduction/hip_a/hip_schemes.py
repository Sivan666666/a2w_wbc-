"""Shared experiment specification; wheel terms are the user's corrected B baseline."""
HIP_WEIGHTS = {
    'a': (-0.3, -1.0, -0.2),
    'b': (-0.15, -0.5, -0.1),
    'c': (-0.03, -0.1, -0.02),
    'd': (-0.3, 0.0, 0.0),
}
WHEEL_ACTUAL_WEIGHT = -0.05
WHEEL_TARGET_WEIGHT = -0.01
HIP_NAMES = [f'{leg}_hip_joint' for leg in ('FR', 'FL', 'RR', 'RL')]

def apply_hip_scheme(cfg, scheme):
    for name, weight in zip(('hip_deviation', 'hip_comfort', 'hip_mirror'), HIP_WEIGHTS[scheme]):
        if weight == 0:
            setattr(cfg.rewards, name, None)
        else:
            getattr(cfg.rewards, name).weight = weight
    return dict(zip(('hip_deviation', 'hip_comfort', 'hip_mirror'), HIP_WEIGHTS[scheme]))
