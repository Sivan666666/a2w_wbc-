"""Wheeled-legged WBC: 18 position actions + four wheel velocity actions.

Retains official B2 WBC task/commands; wheel rolling is not penalized as foot sliding.
Actor has 18 relative positions, 22 velocities, 22 previous actions; wheel phase is omitted.
"""
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ... import mdp
from ..b2_z1.wbc_env_cfg import B2Z1WBCEnvCfg
from ..b2_z1.agents.rsl_rl_ppo_cfg import B2Z1WBCPPORunnerCfg
from LeggedManip_Lab.assets.a2w_z1.a2w_z1_articulation_cfg import (
    A2W_Z1_CFG, POSITION_JOINTS, ACTIVE_JOINTS, WHEELS)

@configclass
class A2WActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=POSITION_JOINTS,
        scale=.25, use_default_offset=True, preserve_order=True, clip={".*":(-10.,10.)})
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=WHEELS,
        scale=10., use_default_offset=False, preserve_order=True, clip={".*":(-3.,3.)})

@configclass
class A2WZ1WBCEnvCfg(B2Z1WBCEnvCfg):
    actions: A2WActionsCfg = A2WActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = A2W_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["joint_names"] = POSITION_JOINTS
            group.joint_vel.params["joint_names"] = ACTIVE_JOINTS
        self.observations.critic.feet_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=".*_wheel")
        # Wheels should roll and normally remain on the floor; do not reward foot swing gait.
        for name in ("feet_slide", "feet_air_time", "feet_long_air", "air_time_variance", "joint_mirror"):
            setattr(self.rewards, name, None)
        self.rewards.track_base_height_exp.params["target_height"] = .48
        # Conservative wheeled command curriculum, same whole-body EE task as B2.
        self.commands.base_velocity.ranges.lin_vel_y = (-.05, .05)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-.2, .2)

@configclass
class A2WZ1WBCEnvCfg_PLAY(A2WZ1WBCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges

@configclass
class A2WZ1WBCPPORunnerCfg(B2Z1WBCPPORunnerCfg):
    experiment_name = "a2w_z1_wbc"
