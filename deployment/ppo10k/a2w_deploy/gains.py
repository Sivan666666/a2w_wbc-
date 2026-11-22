"""Separate preparation and PPO servo gains, expressed in SI units."""
from .policy import KP, KD


def phase_gains(config, phase):
    """Return 22 gains in policy order: FR/FL/RR/RL legs, Z1, wheels."""
    kp, kd = KP.copy(), KD.copy()
    kp[:12] = config[f'{phase}_leg_kp']
    kd[:12] = config[f'{phase}_leg_kd']
    arm_phase = 'prepare' if phase == 'stand' else phase
    kp[12:18] = config[f'{arm_phase}_arm_kp']
    kd[12:18] = config[f'{arm_phase}_arm_kd']
    return kp, kd
