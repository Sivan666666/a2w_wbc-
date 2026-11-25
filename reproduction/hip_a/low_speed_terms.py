import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from low_speed_math import gate,ramp,sample_commands

class LowSpeedMixtureCommand(UniformVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.mode=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.sample_counts=torch.zeros(3,device=self.device,dtype=torch.long)
    def _resample_command(self,env_ids):
        ranges=self.cfg.ranges
        cmd,mode=sample_commands(len(env_ids),self.device,[ranges.lin_vel_x,ranges.lin_vel_y,ranges.ang_vel_z])
        self.vel_command_b[env_ids]=cmd
        self.mode[env_ids]=mode
        self.is_standing_env[env_ids]=mode==0
        self.is_heading_env[env_ids]=False
        self.sample_counts+=torch.bincount(mode,minlength=3)

def low_speed_wheel_actual_l2(env,final_actual,asset_cfg:SceneEntityCfg):
    actual_scale,_=ramp(env.common_step_counter+getattr(env,"training_step_offset",0),final_actual)
    w=env.scene[asset_cfg.name].data.joint_vel[:,asset_cfg.joint_ids]
    return actual_scale*gate(env.command_manager.get_command('base_velocity'))*w.square().sum(dim=-1)

def low_speed_wheel_target_l2(env):
    _,target_scale=ramp(env.common_step_counter+getattr(env,"training_step_offset",0),1.)
    target=env.action_manager.get_term('wheel_vel').processed_actions
    return target_scale*gate(env.command_manager.get_command('base_velocity'))*target.square().sum(dim=-1)
