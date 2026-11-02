"""Simulator-backed preflight of immutable baseline, sampling, reward semantics."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
import copy
import json
from pathlib import Path
import torch
import gymnasium as gym
import LeggedManip_Lab
from isaaclab.utils.math import quat_apply_inverse
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.a2w_z1_wheel_v2.wbc_env_cfg import A2WZ1WheelV2EnvCfg
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.a2w_z1_wheel_v2_precision.wbc_env_cfg import A2WZ1WheelV2PrecisionEnvCfg

old=A2WZ1WheelV2EnvCfg()
before=copy.deepcopy(old.to_dict())
new=A2WZ1WheelV2PrecisionEnvCfg()
assert old.to_dict()==before, 'baseline instance mutated'
assert A2WZ1WheelV2EnvCfg().to_dict()==before, 'baseline defaults mutated'
trim=copy.deepcopy(new.to_dict())
for part in ('ranges','limit_ranges'):
    assert tuple(trim['commands']['ee_pose'][part]['pos_x'])==(.1,.85)
    trim['commands']['ee_pose'][part]['pos_x']=before['commands']['ee_pose'][part]['pos_x']
for key in ('ee_position_medium','ee_position_fine'): trim['rewards'].pop(key)
def differences(x,y,path=''):
    if isinstance(x,dict) and isinstance(y,dict):
        result=[]
        for key in x.keys()|y.keys():
            if key not in x or key not in y: result.append((path+'.'+key,repr(x.get(key)),repr(y.get(key))))
            else: result.extend(differences(x[key],y[key],path+'.'+key))
        return result
    return [] if x==y else [(path,repr(x),repr(y))]
diff=differences(trim,before)
print('CONFIG_DIFF',json.dumps(diff),flush=True)
assert not diff, 'unexpected change outside x-range and two position rewards'

new.scene.num_envs=64
new.sim.device='cuda:0'
new.scene.terrain.visual_material=None
new.scene.sky_light.spawn.texture_file=None
new.commands.ee_pose.debug_vis=False
new.commands.base_velocity.debug_vis=False
env=gym.make('A2W-Z1-WBC-WheelV2-Precision',cfg=new).unwrapped
obs,_=env.reset(seed=42)
term=env.command_manager.get_term('ee_pose')
assert not term.cfg.curriculum_enabled
assert tuple(term.cfg.ranges.pos_x)==(.1,.85), 'reset overwrote requested range'
samples=[]
for _ in range(100):
    term._resample_command(torch.arange(env.num_envs,device=env.device))
    samples.append(term.command[:,:3].clone())
samples=torch.cat(samples)
assert torch.all(samples[:,0]>=.1) and torch.all(samples[:,0]<=.85)
assert float(samples[:,0].min())<.11 and float(samples[:,0].max())>.84
assert torch.all(samples[:,1]>=-.35) and torch.all(samples[:,1]<=.35)
assert torch.all(samples[:,2]>=.1) and torch.all(samples[:,2]<=1.)
assert env.action_manager.total_action_dim==22
assert obs['policy'].shape[-1]==234, obs['policy'].shape

robot=env.scene['robot']
ee=robot.find_bodies('end_effector')[0][0]
link=robot.find_bodies('link0')[0][0]
actual=quat_apply_inverse(robot.data.body_quat_w[:,link],robot.data.body_pos_w[:,ee]-robot.data.body_pos_w[:,link])
values={}
for offset in (0.,.01,.05,.1,.3):
    term.command[:,:2]=actual[:,:2]
    term.command[:,0]+=offset
    term.command[:,2]=robot.data.body_pos_w[:,ee,2]
    values[str(offset)]={}
    for name,std in (('end_effector_position_tracking_exp',.1**.5),('ee_position_medium',.1),('ee_position_fine',.05)):
        cfg=env.reward_manager.get_term_cfg(name)
        r=cfg.func(env,**cfg.params)
        expected=torch.exp(torch.tensor(-offset**2/std**2,device=env.device))
        assert torch.allclose(r,expected.expand_as(r),atol=2e-6)
        values[str(offset)][name]=float(r.mean())*cfg.weight
env.reset()
for _ in range(64):
    obs,rew,terminated,truncated,info=env.step(torch.zeros(env.num_envs,22,device=env.device))
    assert torch.isfinite(rew).all()
    assert torch.isfinite(obs['policy']).all()
report=dict(passed=True,baseline_unchanged=True,allowed_changes=['EE x lower bound','medium/fine position rewards'],
    sampling_count=len(samples),xyz_min=samples.min(0).values.tolist(),xyz_max=samples.max(0).values.tolist(),
    actor_obs_shape=list(obs['policy'].shape),num_actions=env.action_manager.total_action_dim,
    known_position_error_weighted_rewards=values,finite_zero_action_steps=64)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n')
print('PRECISION_AUDIT_PASSED',json.dumps(report))
env.close();app.close()
