"""Frozen LC30K: estimated-state impedance + horizontal tension-only rope.
No simulator state feeds impedance controller. Truth only drives environment rope and diagnostics.
"""
import argparse,json,math,time,traceback,hashlib,sys,xml.etree.ElementTree as ET
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--actor',required=True);p.add_argument('--output',required=True);p.add_argument('--repo',required=True);p.add_argument('--mode',choices=['demo','load'],default='demo');p.add_argument('--render-gpu',type=int,default=0);AppLauncher.add_app_launcher_args(p);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
save('status.json',dict(stage='starting',updated_at=time.time()))
AppLauncher._SIM_APP_CFG_TYPES['create_new_stage']=[bool];a.create_new_stage=False;a.enable_cameras=True;a.active_gpu=a.render_gpu;a.multi_gpu=False
app=AppLauncher(a).app
sys.path.insert(0,str(Path(a.repo)/'reproduction'))
import torch,numpy as np,cv2
import LeggedManip_Lab.tasks
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse,quat_error_magnitude,quat_mul
from force_control.isaac import make_cfg,term,ForceCommand,ee_position_truth
from force_control.core import yaw_angle,yaw_rotate,impedance_external_command
from force_control.stage_eval import EvalPose,EvalVelocity
from isaaclab.markers import VisualizationMarkers,VisualizationMarkersCfg
import isaaclab.sim as sim_utils
clock_t=-5.;enabled=False;anchor0=None;reference=None;estimate=None;vel_est=None;command=None
ROPE_K=150.;ROPE_D=8.;REST=1.;CAP=150.;KP=100.;KD=10.
def demo_extension(t):
 if t<5:return 0.,'BASELINE'
 if t<10:return .15*(t-5)/5,'LIGHT PULL - ramp'
 if t<20:return .15,'LIGHT PULL - hold'
 if t<25:return .15*(25-t)/5,'RELEASE'
 if t<30:return 0.,'RECOVER'
 if t<35:return .30*(t-30)/5,'STRONG PULL - ramp'
 if t<45:return .30,'STRONG PULL - hold'
 if t<50:return .30*(50-t)/5,'RELEASE'
 return 0.,'RECOVER'
def loading(t):
 if a.mode=='demo':return demo_extension(t)
 if t<5:return 0.,'BASELINE'
 # Move the rope handle outward in 0.1m increments: nominal blocked-EE +15N each.
 k=int((t-5)//8)+1;phase=(t-5)%8;extension=.1*(k-1)+.1*min(phase/2,1)
 return extension,f'LOAD {k}: winch target {15*k:.0f}N'
class Pose(EvalPose):
 def _update_command(self):self.pose_command[:]=torch.tensor([.6,0,.5,1.,0,0,0],device=self.device)
class Velocity(EvalVelocity):
 def _update_command(self):self.vel_command_b[:]=0
class Rope(ForceCommand):
 def _resample_command(self,ids):super()._resample_command(ids);self._update_command()
 def _update_command(self):
  self.state.mode[:]=enabled;self.state.free[:]=False;self.state.command[:]=0 if command is None else command
 def apply_force(self):
  d=self.robot.data;f=torch.zeros_like(self.state.external_w)
  self.handle=d.body_pos_w[:,self.ee].clone();self.tension=torch.zeros(1,device=self.device);self.length=self.tension.clone()
  if enabled and anchor0 is not None:
   ext,_=loading(clock_t);self.handle[:]=anchor0;self.handle[:,0]+=ext;self.handle[:,2]=d.body_pos_w[:,self.ee,2]
   delta=self.handle-d.body_pos_w[:,self.ee];delta[:,2]=0
   length=delta.norm(dim=-1).clamp(min=1e-6);direction=delta/length[:,None]
   # Anchor velocity follows prescribed x motion; height follows EE only to constrain force horizontal.
   next_ext,_=loading(clock_t+.001);handle_speed=(next_ext-ext)/.001
   relative=handle_speed*direction[:,0]-(d.body_lin_vel_w[:,self.ee]*direction).sum(-1)
   if a.mode=='load':
    # Ideal tension-controlled winch: pays out as the EE yields, holding target
    # spring extension. Environment truth controls the test fixture ONLY.
    self.handle[:]=d.body_pos_w[:,self.ee];self.handle[:,0]+=REST+ext
    length=torch.full_like(length,REST+ext);direction=torch.zeros_like(direction);direction[:,0]=1;relative=torch.zeros_like(relative)
   self.tension=torch.where(length>REST,(ROPE_K*(length-REST)+ROPE_D*relative).clamp(0,CAP),torch.zeros_like(length))
   f=direction*self.tension[:,None];self.length=length
  self.state.external_w[:]=f;self.state.external_yaw[:]=yaw_rotate(f,yaw_angle(d.root_quat_w),inverse=True)
  flocal=quat_apply_inverse(d.body_quat_w[:,self.ee],f)[:,None,:]
  self.robot.set_external_force_and_torque(flocal,torch.zeros_like(flocal),body_ids=[self.ee]);self.state.applied_substeps+=1
@torch.inference_mode()
def main():
 global clock_t,enabled,anchor0,reference,estimate,vel_est,command
 torch.set_num_threads(4);torch.manual_seed(20261127);np.random.seed(20261127)
 cfg,_=make_cfg(a.device,1);cfg.seed=20261127;cfg.scene.num_envs=1;cfg.episode_length_s=150.;cfg.observations.policy.enable_corruption=False
 cfg.commands.ee_pose.class_type=Pose;cfg.commands.base_velocity.class_type=Velocity;cfg.commands.ee_force.class_type=Rope
 cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None;cfg.viewer.resolution=(960,640);cfg.log_dir=str(out)
 original=Path(cfg.scene.robot.spawn.asset_path);tree=ET.parse(original)
 for mesh in tree.getroot().iter('mesh'):
  path=original.parent/'meshes'/Path(mesh.get('filename')).name;assert path.is_file();mesh.set('filename',str(path))
 tree.write(out/'model.urdf',encoding='utf-8',xml_declaration=True);cfg.scene.robot.spawn.asset_path=str(out/'model.urdf');cfg.scene.robot.spawn.usd_dir=str(out/'usd_sdk_no_ff')
 import omni.usd
 omni.usd.get_context().new_stage()
 for _ in range(5):app.update()
 env=ManagerBasedRLEnv(cfg=cfg,render_mode='rgb_array');assert env.num_envs==1
 robot=env.scene['robot'];pose=env.command_manager.get_term('ee_pose');vel=env.command_manager.get_term('base_velocity');fc=term(env)
 assert type(robot.actuators['arm']).__name__=='DelayedPDActuator'
 actor=torch.jit.load(a.actor,map_location=a.device).eval();obs,_=env.reset(seed=20261127);assert obs['policy'].shape==(1,820)
 marker=VisualizationMarkers(VisualizationMarkersCfg(prim_path='/Visuals/Rope',markers={'rope':sim_utils.SphereCfg(radius=.009,visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.,.55,.05))),'handle':sim_utils.SphereCfg(radius=.045,visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.,.15,.05))),'target':sim_utils.SphereCfg(radius=.025,visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.,.9,1.)))}))
 for i in range(250):obs,_,_,_,_=env.step(actor(obs['policy']))
 enabled=True;fc._update_command();obs=env.observation_manager.compute(update_history=False)
 estimate=actor.estimate(obs['policy']);reference=estimate[:,6:9].clone();previous=reference.clone();vel_est=torch.zeros_like(reference);command=torch.zeros_like(reference)
 anchor0=robot.data.body_pos_w[:,fc.ee].clone();anchor0[:,0]+=REST
 base0=robot.data.root_pos_w.clone();true_ref=ee_position_truth(env).clone();ee0=robot.data.body_pos_w[:,fc.ee].clone();rows=[];pretruth=None;original_compute=env.reward_manager.compute
 arm=robot.find_joints([f'joint{i}' for i in range(1,7)],preserve_order=True)[0]
 legs=robot.find_joints([f'{l}_{j}_joint' for l in ['FR','FL','RR','RL'] for j in ['hip','thigh','calf']],preserve_order=True)[0]
 act=env.action_manager.get_term('joint_pos');ids=robot.find_joints(act.cfg.joint_names,preserve_order=True)[0]
 def capture(dt):
  reward=original_compute(dt);d=robot.data;qtarget=d.joint_pos.clone();qtarget[:,ids]=act.processed_actions
  pos=ee_position_truth(env);q=d.root_quat_w;w,x,y,z=q.unbind(-1);tilt=torch.acos((1-2*(x*x+y*y)).clamp(-1,1))*180/math.pi
  low=d.joint_pos_limits[:,arm,0];high=d.joint_pos_limits[:,arm,1];margin=torch.minimum(d.joint_pos[:,arm]-low,high-d.joint_pos[:,arm])
  vals={'time_s':torch.tensor([clock_t],device=a.device),'ee_world':d.body_pos_w[:,fc.ee],'ee_relative':pos,'reference_estimated':reference,'reference_actual_entry':true_ref,'estimated_position':estimate[:,6:9],'estimated_velocity_filtered':vel_est,'estimated_force':estimate[:,3:6],'force_estimator_truth_pre':pretruth,'force_command_external':command,'force_applied_external':fc.state.external_yaw,'force_applied_world':fc.state.external_w,'rope_tension':fc.tension,'rope_length':fc.length,'handle_world':fc.handle,'base_position':d.root_pos_w,'base_drift':(d.root_pos_w-base0)[:,:2].norm(dim=-1),'base_tilt_deg':tilt,'base_height':d.root_pos_w[:,2],'arm_limit_margin_deg':margin*180/math.pi,'arm_joint_error_deg':(d.joint_pos[:,arm]-qtarget[:,arm])*180/math.pi,'leg_joint_error_deg':(d.joint_pos[:,legs]-qtarget[:,legs])*180/math.pi,'terminated':env.termination_manager.terminated,'base_contact':env.termination_manager.get_term('base_contact'),'bad_orientation':env.termination_manager.get_term('bad_orientation'),'ee_orientation_deg':quat_error_magnitude(d.body_quat_w[:,fc.ee],quat_mul(d.body_quat_w[:,fc.link],pose.command[:,3:]))*180/math.pi}
  rows.append({k:v.detach().cpu().numpy().copy() for k,v in vals.items()});return reward
 env.reward_manager.compute=capture
 writer=cv2.VideoWriter(str(out/'raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(960,640));assert writer.isOpened()
 maxsteps=3000 if a.mode=='demo' else 5050;stop='duration_complete';frames=0
 for i in range(maxsteps):
  clock_t=i*.02
  # Estimate from current proprioceptive history and previous command. No privileged feedback.
  estimate=actor.estimate(obs['policy']);pos_est=estimate[:,6:9];raw_v=(pos_est-previous)/.02;vel_est=.8*vel_est+.2*raw_v;previous=pos_est.clone()
  command=impedance_external_command(pos_est,vel_est,reference,KP,KD,70.)
  pose._update_command();vel._update_command();fc._update_command();obs=env.observation_manager.compute(update_history=False)
  pretruth=fc.state.external_yaw.clone();obs,_,done,timeout,_=env.step(actor(obs['policy']))
  if i%5==0 or bool(done.any()):
   base=robot.data.root_pos_w[0].cpu().numpy();env.sim.set_camera_view(eye=base+np.array([2.,-3.,1.25]),target=base+np.array([.7,0,.25]))
   ee=robot.data.body_pos_w[:,fc.ee];h=fc.handle;u=torch.linspace(0,1,70,device=a.device)[:,None];points=ee+(h-ee)*u
   slack=max(0.,REST-float(fc.length[0]));points[:,2]-=math.sin(0.)+min(.20,slack+.02)*(torch.sin(math.pi*u[:,0]) if float(fc.tension[0])<.1 else 0)
   target_w=robot.data.root_pos_w+yaw_rotate(reference,yaw_angle(robot.data.root_quat_w))
   marker.visualize(translations=torch.cat((points,h,target_w)),marker_indices=torch.tensor([0]*70+[1,2],device=a.device))
   for _ in range(3 if i==0 else 1):env.sim.render();frame=env.render(recompute=True)
   frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR);cv2.rectangle(frame,(0,0),(960,130),(15,20,25),-1)
   last=rows[-1];displacement=float(np.linalg.norm(last['ee_relative']-last['reference_actual_entry']))*100
   lines=[f"LC 30K | ESTIMATED-STATE IMPEDANCE | {a.mode.upper()} | {clock_t:05.1f}s",loading(clock_t)[1],f"Rope tension {float(last['rope_tension'][0]):5.1f} N | target external Fx {float(command[0,0]):+5.1f} N | estimated Fx {float(estimate[0,3]):+5.1f} N",f"EE move {displacement:5.1f} cm | base drift {float(last['base_drift'][0]):.2f} m | tilt {float(last['base_tilt_deg'][0]):.1f} deg",'Orange: tension-only virtual rope | Cyan: body-relative impedance reference']
   for j,line in enumerate(lines):cv2.putText(frame,line,(14,24+j*24),cv2.FONT_HERSHEY_SIMPLEX,.57,(235,235,235),1,cv2.LINE_AA)
   if bool(done.any()):cv2.putText(frame,'TERMINATED - LOAD TEST STOPPED',(140,320),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,0,255),2)
   writer.write(frame);frames+=1
   if i%500==0 or bool(done.any()):cv2.imwrite(str(out/f'preview_{i:04d}.jpg'),frame)
  if i%250==0:save('status.json',dict(stage='recording',step=i,tension=float(rows[-1]['rope_tension'][0]),updated_at=time.time()))
  if bool(done.any()):stop='environment_termination';break
  if float(rows[-1]['base_tilt_deg'][0])>60:stop='tilt_above_60deg';break
  if float(rows[-1]['base_height'][0])<.15:stop='base_height_below_15cm';break
 writer.release();data={k:np.stack([r[k] for r in rows]) for k in rows[0]};assert all(np.isfinite(v).all() for v in data.values());np.savez_compressed(out/'trace.npz',**data)
 save('protocol.json',dict(num_envs=1,seed=20261127,mode=a.mode,frames=frames,duration_s=len(rows)*.02,stop=stop,actor_sha256=hashlib.sha256(Path(a.actor).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kp=KP,kd=KD,velocity_filter='v=.8*v_previous+.2*diff(x_est)/.02',force_command_limit_per_axis_N=70,rope_k_N_m=ROPE_K,rope_d_Ns_m=ROPE_D,rope_rest_m=REST,rope_cap_N=CAP,load_fixture='ideal tension-controlled winch following EE, 15N/8s, 2s ramps, cap150N; fixture feedback uses truth, policy does not',frame='base-yaw axes / base-relative position as trained; derivative in same moving coordinates',reference='one estimated EE position at force-mode entry; constant body-relative target',external_force_only='Horizontal world XY tension from extension and relative velocity, no vertical force. Visual rope has no distributed mass/collision.',formula_external='Kp*(estimated_position-reference)+Kd*estimated_velocity; negative of robot-on-environment spring-damper force',control_privileged_state=False,warmup_s=5,force_mode_entire_recording=True,base_velocity_command=[0,0,0],failures=int(data['terminated'].sum()),max_tension_N=float(data['rope_tension'].max()),max_base_drift_m=float(data['base_drift'].max()),max_base_tilt_deg=float(data['base_tilt_deg'].max())))
 save('status.json',dict(stage='complete',stop=stop,frames=frames,updated_at=time.time()));env.close()
try:main()
except Exception as e:save('status.json',dict(stage='failed',error=str(e),traceback=traceback.format_exc()));raise
finally:app.close()
