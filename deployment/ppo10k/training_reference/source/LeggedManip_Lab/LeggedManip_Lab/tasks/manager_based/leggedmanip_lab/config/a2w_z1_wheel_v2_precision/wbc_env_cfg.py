"""Experimental EE precision variant; not a proven improvement or safety envelope.

Retain original 4.5*exp(-e^2/0.1) capture reward, add
1.0*exp(-e^2/0.01) + 1.5*exp(-e^2/0.0025).
All use the original mixed-frame Euclidean error (link0 XY, world Z).
No change to dynamics, observations, command orientation, base/wheel rewards,
action-rate penalty, physical limits, PPO, or the original WheelV2 class.
"""
from copy import deepcopy
from isaaclab.utils import configclass
from ..a2w_z1_wheel_v2.wbc_env_cfg import A2WZ1WheelV2EnvCfg, A2WZ1WheelV2PPORunnerCfg


@configclass
class A2WZ1WheelV2PrecisionEnvCfg(A2WZ1WheelV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands = deepcopy(self.commands)
        self.commands.ee_pose.ranges.pos_x = (0.10, 0.85)
        self.commands.ee_pose.limit_ranges.pos_x = (0.10, 0.85)
        # configclass.copy() uses dataclasses.replace and drops dynamically
        # attached WheelV2 arm/wheel regularizers. Preserve all actual terms.
        self.rewards = deepcopy(self.rewards)
        term = self.rewards.end_effector_position_tracking_exp
        self.rewards.ee_position_medium = deepcopy(term)
        self.rewards.ee_position_medium.weight = 1.0
        self.rewards.ee_position_medium.params['std'] = 0.10
        self.rewards.ee_position_fine = deepcopy(term)
        self.rewards.ee_position_fine.weight = 1.5
        self.rewards.ee_position_fine.params['std'] = 0.05


@configclass
class A2WZ1WheelV2PrecisionEnvCfg_PLAY(A2WZ1WheelV2PrecisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges


@configclass
class A2WZ1WheelV2PrecisionPPORunnerCfg(A2WZ1WheelV2PPORunnerCfg):
    experiment_name = 'a2w_z1_wbc_wheel_v2_precision'
