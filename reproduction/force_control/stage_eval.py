"""Frozen EE/force protocol. Capture pre-reset state and report modes separately."""
import math, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_error_magnitude
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_wbc import UniformPoseWBCCommand
from .isaac import ForceCommand, term

# name, x, z, pitch_deg, vx, yaw_rate, force_xyz (None -> position mode)
CASES = [
 ('stand',.6,.5,0,0,0,None), ('low_ee',.35,.2,60,0,0,None),
 ('pitch_ee',.65,.5,-45,0,0,None), ('creep',.6,.5,0,.05,0,None),
 ('forward',.6,.5,0,.5,0,None), ('backward',.6,.5,0,-.5,0,None),
 ('turn',.6,.5,0,0,.5,None),
 ('force_zero',.6,.5,0,0,0,(0,0,0)),
 *[(f'force_{axis}_{sign:+d}',.6,.5,0,0,0,tuple(sign*20 if j==i else 0 for j in range(3)))
   for i,axis in enumerate('xyz') for sign in [-1,1]],
 ('force_diagonal_pos',.6,.5,0,0,0,(20,20,20)),
 ('force_diagonal_neg',.6,.5,0,0,0,(-20,-20,-20)),
 ('force_forward',.6,.5,0,.5,0,(20,0,0)),
 ('force_backward',.6,.5,0,-.5,0,(20,0,0)),
 ('force_turn',.6,.5,0,0,.5,(0,20,0)),
]
PER_CASE=4
clock_s=0.

def alpha():return min(max((clock_s-8.)/4.,0.),1.)

class EvalPose(UniformPoseWBCCommand):
 def _resample_command(self,env_ids):self._update_command()
 def _update_command(self):
  a=alpha()
  for j,(_,x,z,pitch,_,_,_) in enumerate(CASES):
   ang=math.radians(pitch)*a
   self.pose_command[j*PER_CASE:(j+1)*PER_CASE]=torch.tensor([.6+(x-.6)*a,0,.5+(z-.5)*a,math.cos(ang/2),0,math.sin(ang/2),0],device=self.device)

class EvalVelocity(UniformVelocityCommand):
 def _resample_command(self,env_ids):self._update_command()
 def _update_command(self):
  for j,(_,_,_,_,vx,yaw,_) in enumerate(CASES):
   self.vel_command_b[j*PER_CASE:(j+1)*PER_CASE]=torch.tensor([vx*alpha(),0,yaw*alpha()],device=self.device)

class EvalForce(ForceCommand):
 def _resample_command(self,env_ids):
  super()._resample_command(env_ids)
  self._update_command()
 def _update_command(self):
  self.state.kp[:]=100.;self.state.kd[:]=10.;self.state.free[:]=False
  for j,case in enumerate(CASES):
   sl=slice(j*PER_CASE,(j+1)*PER_CASE);force=case[-1]
   self.state.mode[sl]=force is not None
   self.state.command[sl]=0 if force is None else torch.tensor(force,device=self.device)*alpha()

def configure(cfg):
 cfg.scene.num_envs=len(CASES)*PER_CASE;cfg.seed=20261127;cfg.episode_length_s=31.
 cfg.commands.ee_pose.class_type=EvalPose;cfg.commands.base_velocity.class_type=EvalVelocity
 cfg.commands.ee_force.class_type=EvalForce
 cfg.observations.policy.enable_corruption=False
 return cfg

def statistics(x):
 x=np.asarray(x)
 return {'bias':np.mean(x,axis=(0,1)).tolist(),'mae':np.mean(np.abs(x),axis=(0,1)).tolist(),
         'rmse':np.sqrt(np.mean(x*x,axis=(0,1))).tolist(),'abs_p95':np.percentile(np.abs(x),95,axis=(0,1)).tolist()}

@torch.inference_mode()
def evaluate(env,model,out,checkpoint,steps=1500):
 global clock_s
 out=Path(out);robot=env.scene['robot'];pose=env.command_manager.get_term('ee_pose');fc=term(env)
 arm=robot.find_joints([f'joint{i}' for i in range(1,7)],preserve_order=True)[0]
 leg_names=[f'{leg}_{joint}_joint' for leg in ['FR','FL','RR','RL'] for joint in ['hip','thigh','calf']]
 legs=robot.find_joints(leg_names,preserve_order=True)[0]
 position_action=env.action_manager.get_term('joint_pos');position_ids=robot.find_joints(position_action.cfg.joint_names,preserve_order=True)[0]
 rows=[];estimate=None;original=env.reward_manager.compute
 def capture(dt):
  reward=original(dt);d=robot.data;ee=d.body_pos_w[:,pose.body_idx];lq=d.body_quat_w[:,pose.link0_idx]
  local=quat_apply_inverse(lq,ee-d.body_pos_w[:,pose.link0_idx]);pc=pose.command
  # Record both historical link0-reference metric and actual reward root-reference metric.
  error_cm=torch.cat((local[:,:2]-pc[:,:2],ee[:,2:3]-pc[:,2:3]),-1)*100
  ori=quat_error_magnitude(d.body_quat_w[:,pose.body_idx],quat_mul(lq,pc[:,3:]))*180/math.pi
  ori_root=quat_error_magnitude(d.body_quat_w[:,pose.body_idx],quat_mul(d.root_quat_w,pc[:,3:]))*180/math.pi
  qtarget=d.joint_pos.clone();qtarget[:,position_ids]=position_action.processed_actions
  vals={'ee_error_xyz_cm':error_cm,'ee_cm':error_cm.norm(dim=-1),'ee_deg':ori,'ee_deg_root_reference':ori_root,
        'force_command_N':fc.state.command,'force_applied_N':fc.state.external_yaw,'force_estimated_N':estimate[:,3:6],
        'force_mode':fc.state.mode,'velocity_command':env.command_manager.get_command('base_velocity'),
        'velocity':torch.cat((d.root_lin_vel_b[:,:2],d.root_ang_vel_b[:,2:3]),-1),
        'arm_joint_error_deg':(d.joint_pos[:,arm]-qtarget[:,arm])*180/math.pi,
        'leg_joint_error_deg':(d.joint_pos[:,legs]-qtarget[:,legs])*180/math.pi,
        'terminated':env.termination_manager.terminated,'timeout':env.termination_manager.time_outs,
        'base_contact':env.termination_manager.get_term('base_contact'),'bad_orientation':env.termination_manager.get_term('bad_orientation')}
  rows.append({k:v.detach().cpu().numpy().copy() for k,v in vals.items()});return reward
 clock_s=0.;obs,_=env.reset(seed=20261127);rows=[];env.reward_manager.compute=capture
 # Log actor's pre-action estimate and paired pre-action truth separately from post-physics tracking.
 pretruth=[]
 try:
  for i in range(steps):
   clock_s=i*env.step_dt
   estimate=model.estimator(obs['policy']);pretruth.append(obs['estimator'][:,3:6].cpu().numpy().copy())
   obs,_,_,_,_=env.step(model(obs['policy']))
   if i%250==0:(out/'status.json').write_text(json.dumps({'stage':'evaluating','control_step':i,'total_steps':steps,'updated_at':time.time()}))
 finally:env.reward_manager.compute=original
 data={k:np.stack([row[k] for row in rows]) for k in rows[0]};data['force_estimator_target_N']=np.stack(pretruth)
 for k,v in data.items():
  if not np.isfinite(v).all():raise FloatingPointError(k)
 assert data['ee_cm'].shape==(steps,len(CASES)*PER_CASE)
 np.savez_compressed(out/'trace.npz',**data)
 results=[]
 for j,case in enumerate(CASES):
  sl=slice(j*PER_CASE,(j+1)*PER_CASE);entry={'case':case[0],'mode':'position' if case[-1] is None else 'force','command':case[1:],
    'failure_events':int(data['terminated'][:,sl].sum()),'failure_base_contact':int(data['base_contact'][:,sl].sum()),'failure_bad_orientation':int(data['bad_orientation'][:,sl].sum())}
  for label,window in [('full',slice(None)),('last5s',slice(-min(250,steps),None))]:
   get=lambda key:data[key][window,sl]
   entry[label]={
    'ee_position_cm':statistics(get('ee_cm')),'ee_orientation_deg_link0':statistics(get('ee_deg')),
    'ee_orientation_deg_root':statistics(get('ee_deg_root_reference')),
    'force_tracking_N':statistics(get('force_applied_N')-get('force_command_N')),
    'force_tracking_vector_norm_N':statistics(np.linalg.norm(get('force_applied_N')-get('force_command_N'),axis=-1)),
    'force_estimation_N':statistics(get('force_estimated_N')-get('force_estimator_target_N')),
    'force_estimation_vector_norm_N':statistics(np.linalg.norm(get('force_estimated_N')-get('force_estimator_target_N'),axis=-1)),
    'base_velocity_error':statistics(get('velocity')-get('velocity_command')),
    'arm_joint_error_deg':statistics(get('arm_joint_error_deg')),'dog_leg_joint_error_deg':statistics(get('leg_joint_error_deg'))}
  results.append(entry)
 summary={'protocol':'7 position + 12 force cases; 4 env each; 30s; seed20261127; 0-8s neutral, 8-12s ramp, then hold; field Kp100/Kd10; force-zero case retains spring; noise off; existing training randomization retained',
  'steps':steps,'num_envs':env.num_envs,'case_count':len(CASES),'finite':True,'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
  'force_frame':'external force ON robot in base-yaw axes; sign opposite robot-on-environment reaction',
  'timing':'force tracking and EE measured pre-reset after physics; force estimate compared with paired pre-action estimator truth',
  'comparison_note':'EE position metrics assess position-mode cases; force-mode EE position is diagnostic only, not a commanded position tracking score. Both orientation references retained explicitly.',
  'cases':results}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary
