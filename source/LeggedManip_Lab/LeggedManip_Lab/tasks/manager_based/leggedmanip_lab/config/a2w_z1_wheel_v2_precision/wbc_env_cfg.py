"""Experimental EE precision variant; not a proven improvement or safety envelope.

Retain original 4.5*exp(-e^2/0.1) capture reward, add
1.0*exp(-e^2/0.01) + 1.5*exp(-e^2/0.0025).
All use the original mixed-frame Euclidean error (link0 XY, world Z).
Hip regularization: home L1 (-0.3), squared excess beyond 0.25 rad (-1.0),
and left/right hip mirror error (-0.2), all active without an orientation gate.
No change to dynamics, observations, command orientation, base/wheel rewards,
action-rate penalty, physical limits, PPO, or the original WheelV2 class.
"""
from copy import deepcopy
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from ... import mdp
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

        hips = [f'{leg}_hip_joint' for leg in ('FR', 'FL', 'RR', 'RL')]
        # Keep the legacy gated joint_deviation_l1 for the arm/thigh/calf terms.
        # These hip-only terms remain active during turns and large body tilts.
        self.rewards.hip_deviation = RewTerm(
            func=mdp.hip_deviation_l1, weight=-0.3,
            params={'asset_cfg': SceneEntityCfg('robot', joint_names=hips, preserve_order=True)},
        )
        self.rewards.hip_comfort = RewTerm(
            func=mdp.hip_comfort_l2, weight=-1.0,
            params={
                'asset_cfg': SceneEntityCfg('robot', joint_names=hips, preserve_order=True),
                'comfort_angle': 0.25,
            },
        )
        self.rewards.hip_mirror = RewTerm(
            func=mdp.hip_mirror_l2, weight=-0.2,
            params={'asset_cfg': SceneEntityCfg('robot', joint_names=hips, preserve_order=True)},
        )


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
