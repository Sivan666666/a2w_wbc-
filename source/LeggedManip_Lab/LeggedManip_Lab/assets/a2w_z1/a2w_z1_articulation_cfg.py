"""A2W+Z1 full dynamic body, adapted from the user's door URDF (not float-IK split)."""
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg

ASSET_DIR = Path(__file__).resolve().parent
LEGS = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
ARM = [f"joint{i}" for i in range(1, 7)]
WHEELS = [f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")]
POSITION_JOINTS = LEGS + ARM
ACTIVE_JOINTS = POSITION_JOINTS + WHEELS

A2W_Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSET_DIR / "a2wz1_wbc.urdf"),
        usd_dir=str(ASSET_DIR / "usd"), usd_file_name="a2wz1_wbc.usd",
        fix_base=False, merge_fixed_joints=False, make_instanceable=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
            solver_position_iteration_count=4, solver_velocity_iteration_count=0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.55),
        joint_pos={".*_hip_joint": 0., ".*_thigh_joint": .8, ".*_calf_joint": -1.5,
                   ".*_wheel_joint": 0., "joint[1-6]": 0.}, joint_vel={".*": 0.}),
    soft_joint_pos_limit_factor=.9,
    actuators={
        "hips_thighs": DelayedPDActuatorCfg(joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            stiffness=120., damping=3., effort_limit=120., effort_limit_sim=120.,
            velocity_limit_sim=22., armature=.01, min_delay=0, max_delay=4),
        "calves": DelayedPDActuatorCfg(joint_names_expr=[".*_calf_joint"],
            stiffness=120., damping=3., effort_limit=180., effort_limit_sim=180.,
            velocity_limit_sim=14.6667, armature=.01, min_delay=0, max_delay=4),
        "wheels": DelayedPDActuatorCfg(joint_names_expr=[".*_wheel_joint"],
            stiffness=0., damping=2., effort_limit=24.5, effort_limit_sim=24.5,
            velocity_limit_sim=114.4, armature=.01, min_delay=0, max_delay=4),
        "arm": DelayedPDActuatorCfg(joint_names_expr=["joint[1-6]"],
            stiffness={"joint1":50., "joint2":50., "joint3":80., "joint4":30., "joint5":30., "joint6":20.},
            damping={"joint1":3., "joint2":2., "joint3":3., "joint4":3., "joint5":2.5, "joint6":1.},
            effort_limit={"joint[13456]":30., "joint2":60.}, effort_limit_sim={"joint[13456]":30., "joint2":60.},
            velocity_limit_sim=10., armature=.01, min_delay=0, max_delay=4),
    },
)
