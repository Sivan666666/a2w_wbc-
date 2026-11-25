"""Evaluate one exported stage in a fresh Isaac process using the current model."""
import argparse,json,math,time,traceback,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--step',type=int,required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();run=Path(a.run);out=run/'evaluation'/f'stage_{a.step:07d}';out.mkdir(parents=True,exist_ok=False)
def save(name,v):(out/name).write_text(json.dumps(v,indent=2))
save('status.json',dict(stage='starting',updated_at=time.time()));app=AppLauncher(a).app
import torch,numpy as np,gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse,quat_mul,quat_error_magnitude
from isaaclab.utils.io import dump_yaml
import LeggedManip_Lab.tasks
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_wbc import UniformPoseWBCCommand
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from hip_schemes import apply_hip_scheme,HIP_NAMES
cases=[('stand',.6,.5,0,0,0),('low_ee',.35,.2,60,0,0),('pitch_ee',.65,.5,-45,0,0),('creep',.6,.5,0,.05,0),('forward',.6,.5,0,.5,0),('backward',.6,.5,0,-.5,0),('turn',.6,.5,0,0,.5)]
clock_t=0.
class EvalPose(UniformPoseWBCCommand):
 def _resample_command(self,env_ids):self._update_command()
 def _update_command(self):
  alpha=min(max((clock_t-8)/4,0),1)
  for j,(_,x,z,pitch,vx,yaw) in enumerate(cases):
   ang=math.radians(pitch)*alpha
   self.pose_command[j*4:(j+1)*4]=torch.tensor([.6+(x-.6)*alpha,0,.5+(z-.5)*alpha,math.cos(ang/2),0,math.sin(ang/2),0],device=self.device)
class EvalVelocity(UniformVelocityCommand):
 def _resample_command(self,env_ids):self._update_command()
 def _update_command(self):
  alpha=min(max((clock_t-8)/4,0),1)
  for j,(_,x,z,pitch,vx,yaw) in enumerate(cases):self.vel_command_b[j*4:(j+1)*4]=torch.tensor([vx*alpha,0,yaw*alpha],device=self.device)
@torch.inference_mode()
def main():
 global clock_t
 torch.set_num_threads(4);task='A2W-Z1-WBC-WheelV2-Precision-Play';cfg=parse_env_cfg(task,device=a.device,num_envs=len(cases)*4);cfg.seed=20261123
 manifest=json.loads((run/'manifest.json').read_text());apply_hip_scheme(cfg,manifest['scheme'])
 cfg.commands.ee_pose.class_type=EvalPose;cfg.commands.base_velocity.class_type=EvalVelocity
 cfg.commands.ee_pose.debug_vis=cfg.commands.base_velocity.debug_vis=False
 cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
 cfg.scene.robot.spawn.usd_dir=str(out/'usd_no_camera_mounts_sdk_limits');cfg.episode_length_s=31.;cfg.log_dir=str(out)
 dump_yaml(str(out/'env.yaml'),cfg)
 env=gym.make(task,cfg=cfg);u=env.unwrapped;robot=u.scene['robot'];pose=u.command_manager.get_term('ee_pose');wi=robot.find_joints([f'{x}_wheel_joint' for x in ['FR','FL','RR','RL']],preserve_order=True)[0]
 hi=robot.find_joints(HIP_NAMES,preserve_order=True)[0]
 sensor=u.scene['contact_forces'];ci=sensor.find_bodies([f'{leg}_wheel' for leg in ['FR','FL','RR','RL']],preserve_order=True)[0]
 rows=[];original=u.reward_manager.compute
 def capture(dt):
  r=original(dt);ee=robot.data.body_pos_w[:,pose.body_idx];lq=robot.data.body_quat_w[:,pose.link0_idx];local=quat_apply_inverse(lq,ee-robot.data.body_pos_w[:,pose.link0_idx]);q=robot.data.root_quat_w
  vals=dict(hip_q=robot.data.joint_pos[:,hi],root_quat=robot.data.root_quat_w,projected_gravity=robot.data.projected_gravity_b,wheel_contact_forces=sensor.data.net_forces_w[:,ci],wheel_contact_forces_history=sensor.data.net_forces_w_history[:,:,ci],base_contact=u.termination_manager.get_term('base_contact'),bad_orientation=u.termination_manager.get_term('bad_orientation'),root_pos=robot.data.root_pos_w,wheel_q=robot.data.joint_pos[:,wi],wheel_velocity=robot.data.joint_vel[:,wi],wheel_target=u.action_manager.get_term('wheel_vel').processed_actions,wheel_torque=robot.data.applied_torque[:,wi],command=u.command_manager.get_command('base_velocity'),velocity=torch.cat([robot.data.root_lin_vel_b[:,:2],robot.data.root_ang_vel_b[:,2:3]],-1),ee_cm=torch.cat([local[:,:2]-pose.command[:,:2],ee[:,2:3]-pose.command[:,2:3]],-1).norm(dim=-1)*100,ee_deg=quat_error_magnitude(robot.data.body_quat_w[:,pose.body_idx],quat_mul(lq,pose.command[:,3:]))*180/math.pi,pitch_deg=torch.asin((2*(q[:,0]*q[:,2]-q[:,3]*q[:,1])).clamp(-1,1))*180/math.pi,terminated=u.termination_manager.terminated)
  rows.append({k:v.detach().cpu().numpy().copy() for k,v in vals.items()});return r
 u.reward_manager.compute=capture;all_summaries={}
 for completed in [a.step]:
  model=run/'stages'/f'actor_{completed:07d}.jit';actor=torch.jit.load(str(model),map_location=a.device).eval();clock_t=0.;obs,_=env.reset(seed=20261123);rows=[]
  with torch.inference_mode():
   for step in range(1500):
    clock_t=(step+1)*u.step_dt;obs,_,_,_,_=env.step(actor(obs['policy']))
    if step%500==0:save('status.json',dict(stage='evaluating',total_training_updates=completed,step=step,updated_at=time.time()))
  d={k:np.stack([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(out/f'trace_{completed:07d}.npz',**d);results=[]
  for j,c in enumerate(cases):
   sl=slice(j*4,(j+1)*4);late=slice(-250,None);r=dict(case=c[0],failures=int(d['terminated'][:,sl].sum()))
   for key in ['wheel_velocity','wheel_target','wheel_torque','velocity','ee_cm','ee_deg','pitch_deg']:
    v=d[key][late,sl];r[key+'_mean']=v.mean((0,1)).tolist();r[key+'_abs_mean']=np.abs(v).mean((0,1)).tolist()
   hips=d['hip_q'][late,sl]
   r['hip_mean_deg']=np.rad2deg(hips).mean((0,1)).tolist();r['hip_abs_mean_deg']=np.rad2deg(np.abs(hips)).mean((0,1)).tolist();r['hip_abs_max_deg']=np.rad2deg(np.abs(hips)).max((0,1)).tolist()
   r['hip_outside_comfort_fraction']=(np.abs(hips)>.25).mean((0,1)).tolist()
   contact=np.linalg.norm(d['wheel_contact_forces_history'][late,sl],axis=-1).max(axis=2)>1.0
   r['wheel_contact_fraction']=contact.mean((0,1)).tolist()
   r['failure_base_contact']=int(d['base_contact'][:,sl].sum());r['failure_bad_orientation']=int(d['bad_orientation'][:,sl].sum())
   r['ee_cm_p95']=float(np.percentile(d['ee_cm'][late,sl],95));r['ee_deg_p95']=float(np.percentile(d['ee_deg'][late,sl],95))
   r['velocity_target_mean']=d['command'][late,sl].mean((0,1)).tolist();r['velocity_error_bias']=(d['velocity'][late,sl]-d['command'][late,sl]).mean((0,1)).tolist()
   r['velocity_error_rms']=np.sqrt(np.mean((d['velocity'][late,sl]-d['command'][late,sl])**2,axis=(0,1))).tolist()
   r['wheel_net_angle_deg']=np.rad2deg(d['wheel_q'][-1,sl]-d['wheel_q'][-250,sl]).tolist()
   r['wheel_angle_rate_abs_mean']=np.abs(np.diff(d['wheel_q'][late,sl],axis=0)/u.step_dt).mean((0,1)).tolist()
   r['root_xy_displacement_m']=(d['root_pos'][-1,sl,:2]-d['root_pos'][-250,sl,:2]).tolist()
   # Drop intervals crossing a reset: reset jumps are not physical wheel rotation or drift.
   for label,window in [('full',slice(None)),('late',late)]:
    valid=~d['terminated'][window,sl][:-1].astype(bool)
    dq=np.diff(d['wheel_q'][window,sl],axis=0)
    dp=np.diff(d['root_pos'][window,sl,:2],axis=0)
    r[label+'_wheel_cumulative_abs_rotation_deg']=np.rad2deg((np.abs(dq)*valid[:,:,None]).sum(axis=0)).tolist()
    r[label+'_wheel_angle_rate_abs_mean_no_reset']=(np.abs(dq)*valid[:,:,None]).sum((0,1)).tolist()
    r[label+'_wheel_angle_rate_abs_mean_no_reset']=[x/max(int(valid.sum()),1)/u.step_dt for x in r[label+'_wheel_angle_rate_abs_mean_no_reset']]
    r[label+'_root_xy_path_m']=(np.linalg.norm(dp,axis=-1)*valid).sum(axis=0).tolist()
    r[label+'_valid_motion_intervals']=int(valid.sum())
   results.append(r)
  all_summaries[str(completed)]=dict(actor_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),cases=results)
  save('summary.json',dict(protocol='Original training PD/no FF; noise off; 7 cases x 4 envs, seed20261123, same reset seed per checkpoint, native PLAY DR retained; 30s each, final5s metrics; full-trace failure count. Selected hip scheme; no-camera-mount geometry with SDK-aligned limits; each checkpoint evaluated in a fresh process with seed20261123; ordinary evaluation commands, no low-speed mixture override. Added hip angles, wheel contact history, root orientation and termination causes.',stages=all_summaries));print('EVAL COMPLETE',completed,flush=True)
 save('status.json',dict(stage='complete',updated_at=time.time()));env.close()
try:main()
except Exception as exc:
 save('status.json',dict(stage='failed',error=str(exc),traceback=traceback.format_exc(),updated_at=time.time()));traceback.print_exc();raise
finally:app.close()
