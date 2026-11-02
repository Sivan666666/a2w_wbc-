"""A2W+Z1 WheelV2: robot_lab-inspired wheel regularization, original EE task.

Reference robot_lab commit: 500399ed75f510aeaff28705a8ce736c514dbec3.
User-selected wheels: scale 5, target +/-100 rad/s, sim limit 50 rad/s,
Kp/Kd 0/0.6, torque 20 Nm. Retains DelayedPD and original delay randomization.
Legacy checkpoints must use A2W-Z1-WBC[-Play], NOT this new task.
"""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ..a2w_z1.wbc_env_cfg import A2WZ1WBCEnvCfg, A2WZ1WBCPPORunnerCfg
from LeggedManip_Lab.assets.a2w_z1.a2w_z1_articulation_cfg import LEGS, ARM, WHEELS


@configclass
class A2WZ1WheelV2EnvCfg(A2WZ1WBCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Explicit copies: never mutate legacy configs or the shared asset template.
        self.scene.robot = self.scene.robot.copy()
        wheel = self.scene.robot.actuators["wheels"]
        wheel.stiffness = 0.0
        wheel.damping = 0.6
        wheel.effort_limit = 20.0
        wheel.effort_limit_sim = 20.0
        wheel.velocity_limit_sim = 50.0
        self.actions = self.actions.copy()
        self.actions.wheel_vel.scale = 5.0
        self.actions.wheel_vel.clip = {".*": (-100.0, 100.0)}

        self.rewards = self.rewards.copy()
        # Same leg/arm acceleration penalty as before; wheels 100x weaker.
        self.rewards.dof_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2, weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)},
        )
        self.rewards.wheel_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2, weight=-2.5e-9,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEELS)},
        )
        self.rewards.arm_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2, weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM)},
        )
        # No wheel torque or wheel-speed-squared penalty. Preserve arm regularization.
        self.rewards.dof_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2, weight=-2e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)},
        )
        self.rewards.arm_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2, weight=-2e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM)},
        )
        self.rewards.joint_power = RewTerm(
            func=mdp.joint_power, weight=-2e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)},
        )
        for name in ("feet_slide", "feet_air_time", "feet_long_air", "air_time_variance", "joint_mirror"):
            setattr(self.rewards, name, None)
        # EE rewards, commands, curricula, base stability and action smoothness
        # intentionally unchanged: this remains a WBC task, not locomotion-only.


@configclass
class A2WZ1WheelV2EnvCfg_PLAY(A2WZ1WheelV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges


@configclass
class A2WZ1WheelV2PPORunnerCfg(A2WZ1WBCPPORunnerCfg):
    experiment_name = "a2w_z1_wbc_wheel_v2"
