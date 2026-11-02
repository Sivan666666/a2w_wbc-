"""Identical PhysX closed-loop evaluation for BPTT and PPO exported actors."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--actor',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--task',default='A2W-Z1-WBC-WheelV2-Precision-Play')
p.add_argument('--num_envs',type=int,default=64)
p.add_argument('--seed',type=int,default=20261111)
p.add_argument('--seconds',type=float,default=200)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import LeggedManip_Lab.tasks
from supermimic.adapter import Adapter
from supermimic.provenance import capture
main_env=None


def main():
    global main_env
    capture(a.output)
    torch.set_num_threads(4)
    cfg=parse_env_cfg(a.task,device=a.device,num_envs=a.num_envs);cfg.seed=a.seed
    cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
    cfg.commands.ee_pose.debug_vis=cfg.commands.base_velocity.debug_vis=False
    cfg.observations.policy.enable_corruption=False
    env=gym.make(a.task,cfg=cfg);main_env=env;obs,_=env.reset();e=env.unwrapped
    bridge=Adapter(e);policy=torch.jit.load(str(a.actor),map_location=a.device).eval()
    values=[];failed=0;timeout=0
    with torch.no_grad():
        for _ in range(round(a.seconds/e.step_dt)):
            commands=bridge.commands();previous=e.action_manager.action.clone()
            action=policy(obs['policy'])
            obs,_,term,trunc,_=env.step(action)
            proxy=bridge.proxy(bridge.captured['target'],action,previous,commands)
            d=proxy.scene['robot'].data
            from supermimic.core import qapply,qinv,qmul
            cmd=commands[:,3:]
            rel=qapply(qinv(d.body_quat_w[:,bridge.link0]),d.body_pos_w[:,bridge.ee]-d.body_pos_w[:,bridge.link0])
            native=torch.cat((rel[:,:2]-cmd[:,:2],d.body_pos_w[:,bridge.ee,2:3]-cmd[:,2:3]),-1).norm(dim=-1)
            target=d.body_pos_w[:,bridge.link0]+qapply(d.body_quat_w[:,bridge.link0],torch.cat((cmd[:,:2],torch.zeros_like(cmd[:,:1])),-1))
            target=torch.cat((target[:,:2],cmd[:,2:3]),-1)
            world=(target-d.body_pos_w[:,bridge.ee]).norm(dim=-1)
            from isaaclab.utils.math import quat_error_magnitude
            ori=quat_error_magnitude(d.body_quat_w[:,bridge.ee],qmul(d.root_quat_w,cmd[:,3:]))
            xy=(d.root_lin_vel_b[:,:2]-commands[:,:2]).norm(dim=-1)
            yaw=(d.root_ang_vel_b[:,2]-commands[:,2]).abs()
            values.append(torch.stack((native,world,ori,xy,yaw),-1).cpu())
            failed+=int(term.sum());timeout+=int(trunc.sum())
    values=torch.stack(values).flatten(0,1)
    result=dict(actor=str(a.actor),seed=a.seed,num_envs=a.num_envs,seconds=a.seconds,
                failure_resets=failed,timeout_resets=timeout,samples=len(values),noise=False,
                note='Pre-reset physical states; no learned-world-model evaluation; all frames included.')
    for i,name in enumerate(('ee_native_m','ee_world_m','orientation_rad','vel_xy_mps','vel_yaw_rps')):
        result[name]=float(values[:,i].mean());result[name+'_p95']=float(values[:,i].quantile(.95))
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    env.close()

try: main()
finally:
    if main_env is not None: main_env.close()
    app.close()
