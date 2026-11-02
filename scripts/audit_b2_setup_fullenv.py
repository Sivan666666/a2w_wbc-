"""Isolated B2 setup audit through original observation/action managers."""
import argparse
from pathlib import Path
import json, hashlib
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--policy',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse
import LeggedManip_Lab.tasks

a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
policy=torch.jit.load(str(a.policy),map_location='cpu').eval()
ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
print('CHECKPOINT_KEYS',list(ckpt),flush=True)
print('EXPORTED_KEYS',list(policy.state_dict()),flush=True)
matches=[]
for name,v in policy.state_dict().items():
    found=[]
    for group,values in ckpt.items():
        if isinstance(values,dict):
            for k,w in values.items():
                if isinstance(w,torch.Tensor) and w.shape==v.shape and torch.equal(w,v):found.append(group+'.'+k)
    matches.append(dict(export_name=name,shape=list(v.shape),matching_checkpoint_keys=found))
weights_ok=all(v['matching_checkpoint_keys'] for v in matches)
assert weights_ok,matches
model_report=dict(checkpoint_keys=list(ckpt),exported_code=str(policy.code),matches=matches,
    policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),
    checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),iteration=ckpt.get('iter'))
(a.output/'model_identity.json').write_text(json.dumps(model_report,indent=2)+'\n')
print('MODEL_WEIGHTS_MATCH',weights_ok,flush=True)

targets=[[.924,0.,.736],[.8,0.,.736],[.7,0.,.736],[.924,0.,.60],[.8,0.,.60],[.7,0.,.60],[.924,0.,.85],[.8,0.,.85]]
N=len(targets)
cfg=parse_env_cfg('B2-Z1-WBC-Play',device='cuda:0',num_envs=N)
cfg.observations.policy.enable_corruption=False
# Only this process: nominal robot, no startup/reset randomization or imposed pushes.
for name in list(vars(cfg.events)):
    if not name.startswith('_'):setattr(cfg.events,name,None)
for ac in cfg.scene.robot.actuators.values():ac.min_delay=0;ac.max_delay=0
cfg.commands.ee_pose.debug_vis=False
cfg.commands.base_velocity.debug_vis=False
env=gym.make('B2-Z1-WBC-Play',cfg=cfg).unwrapped
env.reset()
robot=env.scene['robot']
names=[f'{l}_{j}_joint' for l in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]+[f'joint{i}' for i in range(1,7)]
ids,_=robot.find_joints(names,preserve_order=True)
link=robot.find_bodies('link0')[0][0];ee=robot.find_bodies('end_effector')[0][0]
origins=env.scene.env_origins
def tensor(x):return torch.tensor(x,dtype=torch.float32,device=robot.device)
root=robot.data.default_root_state.clone();root[:,:3]=origins+tensor([0.,0.,.55]);root[:,3:7]=tensor([1.,0.,0.,0.]);root[:,7:]=0
robot.write_root_state_to_sim(root)
robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.default_joint_vel))
robot.reset();env.action_manager.reset();env.observation_manager.reset()
env.sim.forward();robot.update(.005)
goal=tensor(targets)+origins
history=[np.zeros((N,3,n),np.float32) for n in (3,3,18,18,18,3,7)]
last=np.zeros((N,18),np.float32)
obs_errors=[];action_errors=[];rows=[];geometry=[]
terms=env.observation_manager.active_terms['policy']
assert terms==['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','velocity_commands','pos_commands'],terms
action_term=env.action_manager.get_term('joint_pos')
assert action_term._joint_names==names,action_term._joint_names
for step in range(1000):
    link_pos=robot.data.body_pos_w[:,link];link_quat=robot.data.body_quat_w[:,link]
    delta=quat_apply_inverse(link_quat,goal-link_pos)
    pitch=torch.clamp(-torch.atan2(delta[:,2],torch.linalg.vector_norm(delta[:,:2],dim=1)),-3.14/4,3.14/3)
    yaw=torch.clamp(torch.atan2(delta[:,1],delta[:,0]),-3.14/4,3.14/3)
    cp,sp,cy,sy=torch.cos(pitch/2),torch.sin(pitch/2),torch.cos(yaw/2),torch.sin(yaw/2)
    pose=torch.stack([delta[:,0],delta[:,1],goal[:,2],cp*cy,-sp*sy,sp*cy,cp*sy],dim=1)
    env.command_manager.get_term('ee_pose').pose_command[:]=pose
    env.command_manager.get_command('base_velocity')[:]=0
    observations=env.observation_manager.compute_group('policy',update_history=True)
    raw=[robot.data.root_ang_vel_b.cpu().numpy()*.2,robot.data.projected_gravity_b.cpu().numpy(),
         (robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids]).cpu().numpy(),
         robot.data.joint_vel[:,ids].cpu().numpy()*.05,last,np.zeros((N,3),np.float32),pose.cpu().numpy()]
    for h,v in zip(history,raw):
        if step==0:h[:]=v[:,None,:]
        else:h[:,:-1],h[:,-1]=h[:,1:].copy(),v
    manual=np.concatenate([h.reshape(N,-1) for h in history],axis=1)
    obs_errors.append(float(np.abs(manual-observations.cpu().numpy()).max()))
    with torch.inference_mode():
        action=policy(observations.cpu())
    env.action_manager.process_action(action.to(robot.device))
    expected=torch.clamp(robot.data.default_joint_pos[:,ids]+.25*action.to(robot.device),-10,10)
    action_errors.append(float((expected-action_term.processed_actions).abs().max()))
    actual=robot.data.body_pos_w[:,ee]
    rows.append(dict(t=step*.02,goals=(goal-origins).cpu().tolist(),actual=(actual-origins).cpu().tolist(),
        error_m=torch.linalg.vector_norm(actual-goal,dim=1).cpu().tolist(),
        root_pos=(robot.data.root_pos_w-origins).cpu().tolist(),
        root_quat=robot.data.root_quat_w.cpu().tolist(),
        pose_command=pose.cpu().tolist(),joint_pos=robot.data.joint_pos[:,ids].cpu().tolist(),
        joint_target=expected.cpu().tolist(),torque=robot.data.applied_torque[:,ids].cpu().tolist()))
    if step in (0,999):
        geometry.append(dict(t=step*.02,mount_relative=quat_apply_inverse(robot.data.root_quat_w,link_pos-robot.data.root_pos_w).cpu().tolist(),
            root_quat=robot.data.root_quat_w.cpu().tolist(),link0_quat=link_quat.cpu().tolist()))
    for _ in range(cfg.decimation):
        env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(cfg.sim.dt)
    last=action.numpy()
    if step%250==0:print('HOLD_STEP',step,'ERR',rows[-1]['error_m'],flush=True)
result=dict(observation_max_abs=max(obs_errors),action_target_max_abs=max(action_errors),
    initialization_note='Official CircularBuffer repeats first sample; prior custom demo zero-padded first two history slots.',
    observation_terms=terms,action_joint_names=action_term._joint_names,geometry=geometry,
    limits=robot.data.joint_pos_limits[:,ids].cpu().tolist(),
    setup=dict(dt=cfg.sim.dt,decimation=cfg.decimation,robot_cfg=str(cfg.scene.robot),targets=targets),targets=[])
for i,target in enumerate(targets):
    settled=[r for r in rows if r['t']>=15.]
    diff=np.array([np.array(r['actual'][i])-target for r in settled])
    pos=np.array([r['actual'][i] for r in settled])
    result['targets'].append(dict(goal=target,mean_cm=float(np.linalg.norm(diff,axis=1).mean()*100),
        signed_xyz_cm=(diff.mean(0)*100).tolist(),actual_std_xyz_cm=(pos.std(0)*100).tolist(),
        base_mean=np.mean([r['root_pos'][i] for r in settled],axis=0).tolist()))
(a.output/'hold_trace.json').write_text(json.dumps(rows)+'\n')
(a.output/'setup_results.json').write_text(json.dumps(result,indent=2)+'\n')
print('SETUP_RESULT',json.dumps({k:v for k,v in result.items() if k not in ('setup','limits','geometry')}),flush=True)
env.close()
app.close()
