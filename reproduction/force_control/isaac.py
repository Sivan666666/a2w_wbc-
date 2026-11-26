"""Isaac Lab adapter; existing A configuration is extended only in this opt-in path."""
import copy,sys
from pathlib import Path
import torch
from isaaclab.managers import CommandTerm,CommandTermCfg,ObservationGroupCfg,ObservationTermCfg,RewardTermCfg,SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from isaaclab.envs.mdp.actions import JointPositionAction
from .core import ForceConfig,ForceState,yaw_angle,yaw_rotate,force_reward


class ForceCommand(CommandTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.robot=env.scene['robot'];self.ee=self.robot.find_bodies('end_effector')[0][0]
        self.link=self.robot.find_bodies('link0')[0][0]
        self.state=ForceState(self.num_envs,self.device,ForceConfig(**cfg.force_options))
        self.metrics['force_error_N']=torch.zeros(self.num_envs,device=self.device)
        self.override=None

    @property
    def command(self):
        s=self.state
        return torch.cat((s.command*s.cfg.command_scale,s.mode[:,None].float()),-1)

    def _resample_command(self,env_ids):
        d=self.robot.data
        self.state.reset(env_ids,d.body_pos_w[:,self.ee],d.root_pos_w,d.root_quat_w)
        # Clear stale applied wrench immediately, including on partial resets.
        z=torch.zeros(len(env_ids),1,3,device=self.device)
        self.robot.set_external_force_and_torque(z,z,body_ids=[self.ee],env_ids=env_ids)

    def _update_command(self):
        self.state.advance(self._env.step_dt)
        if self.override is not None:
            self.state.mode[:]=self.override['mode']
            self.state.free[:]=self.override['free']
            self.state.command[:]=self.override['command']*self.state.mode[:,None]

    def _update_metrics(self):
        self.metrics['force_error_N'][:]=torch.linalg.vector_norm(self.state.external_yaw-self.state.command,dim=-1)*self.state.mode

    def anchor_world(self):
        d=self.robot.data
        if self.state.cfg.anchor_mode=='entry_pose':
            return d.root_pos_w+yaw_rotate(self.state.anchor_yaw,yaw_angle(d.root_quat_w))
        # Preserve our original link0 XY/world Z command convention. Solve local Z.
        p=self._env.command_manager.get_command('ee_pose')
        origin=d.body_pos_w[:,self.link];q=d.body_quat_w[:,self.link]
        local=p[:,:3].clone();local[:,2]=0
        base=origin+quat_apply(q,local)
        axis=torch.zeros_like(base);axis[:,2]=1;axis=quat_apply(q,axis)
        denom=axis[:,2]
        # Bounded singularity handling only for the synthetic field; command ABI unchanged.
        denom=torch.where(denom>=0,denom.clamp(min=.05),denom.clamp(max=-.05))
        return base+axis*((p[:,2]-base[:,2])/denom)[:,None]

    def apply_force(self):
        d=self.robot.data
        f=self.state.apply(self.anchor_world(),d.body_pos_w[:,self.ee],d.body_lin_vel_w[:,self.ee],d.root_quat_w)
        # Local per-body wrench preserves other bodies' existing local disturbance buffers.
        flocal=quat_apply_inverse(d.body_quat_w[:,self.ee],f)[:,None,:]
        self.robot.set_external_force_and_torque(flocal,torch.zeros_like(flocal),body_ids=[self.ee])


@configclass
class ForceCommandCfg(CommandTermCfg):
    class_type:type=ForceCommand
    resampling_time_range:tuple=(10.,10.)
    force_options:dict={}
    debug_vis:bool=False


class ForcePositionAction(JointPositionAction):
    def apply_actions(self):
        super().apply_actions()
        term(self._env).apply_force()  # every physics substep, before scene.write_data_to_sim


def term(env):return env.command_manager.get_term('ee_force')

def force_command_observation(env):return term(env).command

def masked_pose_command(env):
    p=env.command_manager.get_command('ee_pose').clone()
    p[:,:3]*=(~term(env).state.mode)[:,None]
    return p  # retain our orientation command as upstream maintain_ori_force_envs=True

def external_force_truth(env):return term(env).state.external_yaw.clone()

def ee_position_truth(env):
    c=term(env);d=c.robot.data
    return yaw_rotate(d.body_pos_w[:,c.ee]-d.root_pos_w,yaw_angle(d.root_quat_w),inverse=True)

def estimator_truth(env):
    return torch.cat((env.scene['robot'].data.root_lin_vel_b,external_force_truth(env),ee_position_truth(env)),-1)

def axis_force_reward(env,axis:int):
    s=term(env).state
    return force_reward(s.external_yaw,s.command,s.mode,s.cfg.force_sigma)[:,axis]

def position_mode_reward(env,command_name:str,std:float,asset_cfg:SceneEntityCfg,link0_name:str='link0'):
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.rewards import position_command_error_exp
    return position_command_error_exp(env,command_name,std,asset_cfg,link0_name)*(~term(env).state.mode)

@configclass
class EstimatorTruthCfg(ObservationGroupCfg):
    state=ObservationTermCfg(func=estimator_truth)
    def __post_init__(self):self.enable_corruption=False;self.concatenate_terms=True


def reward_spec(cfg):
    return {name:{'weight':v.weight,'func':v.func.__module__+':'+v.func.__name__,'params':repr(v.params)}
            for name,v in vars(cfg.rewards).items() if isinstance(v,RewardTermCfg)}


def base_a_cfg(device,num_envs):
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.actuators import DelayedPDActuatorCfg
    from LeggedManip_Lab.assets.a2w_z1.a2w_z1_articulation_cfg import WHEELS
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hip_a'))
    from hip_schemes import apply_hip_scheme
    from low_speed_terms import LowSpeedMixtureCommand,low_speed_wheel_actual_l2,low_speed_wheel_target_l2
    cfg=parse_env_cfg('A2W-Z1-WBC-WheelV2-Precision',device=device,num_envs=num_envs)
    apply_hip_scheme(cfg,'a')
    cfg.commands.base_velocity.class_type=LowSpeedMixtureCommand
    cfg.commands.base_velocity.rel_standing_envs=0.;cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges=copy.deepcopy(cfg.commands.base_velocity.limit_ranges)
    cfg.commands.ee_pose.ranges=copy.deepcopy(cfg.commands.ee_pose.limit_ranges)
    cfg.commands.base_velocity.curriculum_enabled=False;cfg.commands.ee_pose.curriculum_enabled=False
    cfg.commands.base_velocity.debug_vis=False;cfg.commands.ee_pose.debug_vis=False
    cfg.rewards.low_speed_wheel_actual=RewardTermCfg(func=low_speed_wheel_actual_l2,weight=-.05,params={'final_actual':.05,'asset_cfg':SceneEntityCfg('robot',joint_names=WHEELS,preserve_order=True)})
    cfg.rewards.low_speed_wheel_target=RewardTermCfg(func=low_speed_wheel_target_l2,weight=-.01)
    assert type(cfg.scene.robot.actuators['arm']) is DelayedPDActuatorCfg, 'No SDK gravity or residual actuator allowed'
    assert not cfg.scene.robot.spawn.rigid_props.disable_gravity
    return cfg


def make_cfg(device,num_envs,force_cfg=None):
    from dataclasses import asdict
    fc=force_cfg or ForceConfig();cfg=base_a_cfg(device,num_envs);before=reward_spec(cfg)
    cfg.commands.ee_force=ForceCommandCfg(force_options=asdict(fc),resampling_time_range=(fc.mode_interval_s,fc.mode_interval_s))
    cfg.actions.joint_pos.class_type=ForcePositionAction
    cfg.observations.policy.history_length=fc.history_length
    cfg.observations.policy.pos_commands.func=masked_pose_command;cfg.observations.policy.pos_commands.params={}
    cfg.observations.policy.ee_force=ObservationTermCfg(func=force_command_observation)
    cfg.observations.critic.ee_force_command=ObservationTermCfg(func=force_command_observation)
    cfg.observations.critic.ee_external_force=ObservationTermCfg(func=external_force_truth)
    cfg.observations.estimator=EstimatorTruthCfg()
    for name in ('end_effector_position_tracking_exp','ee_position_medium','ee_position_fine'):
        getattr(cfg.rewards,name).func=position_mode_reward
    for i,axis in enumerate('xyz'):
        setattr(cfg.rewards,'ee_force_'+axis,RewardTermCfg(func=axis_force_reward,weight=fc.axis_reward_weight,params={'axis':i}))
    after=reward_spec(cfg)
    gated={'end_effector_position_tracking_exp','ee_position_medium','ee_position_fine'}
    for name,old in before.items():
        assert old['weight']==after[name]['weight'] and old['params']==after[name]['params'],name
        if name not in gated:assert old==after[name],name
    return cfg,{'base':'hip A + SDK software limits + no camera brackets + native delayed PD, NO gravity compensation','before':before,'after':after,'position_mode_gates':sorted(gated),'nonforce_reward_weights_and_params_unchanged':True}
