"""Resume hip A 10K to total50K; unchanged rewards; current no-camera-mount geometry and SDK-aligned limits."""
import argparse,json,time,hashlib,traceback,copy,os,subprocess,sys
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--scheme',choices=['a','b','c','d'],required=True);p.add_argument('--num_envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=40000);p.add_argument('--seed',type=int,default=42);AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def save(name,value):
 path=out/name;temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
save('status.json',dict(stage='starting',updated_at=time.time()));app=AppLauncher(a).app
import torch,numpy as np,gymnasium as gym,importlib.metadata
from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from isaaclab.managers import RewardTermCfg,SceneEntityCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks
from low_speed_terms import LowSpeedMixtureCommand,low_speed_wheel_actual_l2,low_speed_wheel_target_l2
from low_speed_math import gate,ramp
from hip_schemes import apply_hip_scheme,HIP_NAMES,WHEEL_ACTUAL_WEIGHT,WHEEL_TARGET_WEIGHT

def main():
 torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 task='A2W-Z1-WBC-WheelV2-Precision'
 cfg=parse_env_cfg(task,device=a.device,num_envs=a.num_envs);cfg.seed=a.seed
 hip_weights=apply_hip_scheme(cfg,a.scheme)
 actual=-WHEEL_ACTUAL_WEIGHT;target=-WHEEL_TARGET_WEIGHT
 cfg.commands.base_velocity.class_type=LowSpeedMixtureCommand
 cfg.commands.base_velocity.rel_standing_envs=0.;cfg.commands.base_velocity.heading_command=False
 cfg.commands.base_velocity.ranges=copy.deepcopy(cfg.commands.base_velocity.limit_ranges)
 cfg.commands.ee_pose.ranges=copy.deepcopy(cfg.commands.ee_pose.limit_ranges)
 cfg.commands.base_velocity.curriculum_enabled=False;cfg.commands.ee_pose.curriculum_enabled=False
 cfg.commands.base_velocity.debug_vis=False;cfg.commands.ee_pose.debug_vis=False
 cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
 cfg.scene.robot.spawn.usd_dir=str(out/'usd_no_camera_mounts_sdk_limits');cfg.log_dir=str(out)
 wheels=[f'{leg}_wheel_joint' for leg in ['FR','FL','RR','RL']]
 if actual:
  cfg.rewards.low_speed_wheel_actual=RewardTermCfg(func=low_speed_wheel_actual_l2,weight=-actual,params={'final_actual':actual,'asset_cfg':SceneEntityCfg('robot',joint_names=wheels,preserve_order=True)})
 if target:
  cfg.rewards.low_speed_wheel_target=RewardTermCfg(func=low_speed_wheel_target_l2,weight=-target)
 agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point');agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
 agent.seed=a.seed;agent.device=a.device;agent.max_iterations=50000;agent.save_interval=500;agent.logger='tensorboard'
 dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
 env=RslRlVecEnvWrapper(gym.make(task,cfg=cfg),clip_actions=agent.clip_actions);u=env.unwrapped
 class StageRunner(OnPolicyRunner):
  def save(self,path,infos=None):
   # Explicit completed-update stage files avoid RSL's zero-based filenames.
   if Path(path).name.startswith('checkpoint_'):super().save(path,infos)
 runner=StageRunner(env,agent.to_dict(),log_dir=str(out/'tensorboard'),device=a.device)
 runner.load(a.checkpoint,strict=True,map_location=a.device)
 base_iteration=runner.current_learning_iteration
 assert base_iteration==10000 and a.iterations==40000 and a.scheme=='a'
 initial_lr=float(runner.alg.optimizer.param_groups[0]['lr']);runner.alg.learning_rate=initial_lr
 assert len(runner.alg.optimizer.state)>0
 # Restore fidelity: compare every saved algorithm state, including optimizer and normalizers.
 reference=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 restored=runner.alg.save()
 def equal_tree(x,y):
  if torch.is_tensor(x):return torch.is_tensor(y) and torch.equal(x.cpu(),y.cpu())
  if isinstance(x,dict):return set(x)==set(y) and all(equal_tree(x[k],y[k]) for k in x)
  if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal_tree(v,w) for v,w in zip(x,y))
  return x==y
 checks={k:equal_tree(reference[k],v) for k,v in restored.items()}
 assert all(checks.values()),checks
 save('resume_state_parity.json',dict(checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),keys=checks,all_exact=True,base_iteration=base_iteration))
 del reference,restored
 u.training_step_offset=base_iteration*agent.num_steps_per_env
 (out/'stages').mkdir(exist_ok=True)
 runner.export_policy_to_jit(str(out/'stages'),'actor_0010000.jit');runner.alg.train_mode()
 term=u.command_manager.get_term('base_velocity')
 # Count command occupancy at reward time, before reset/resampling. GPU counters only.
 frame_counts=torch.zeros(3,device=a.device,dtype=torch.long)
 original_reward_compute=u.reward_manager.compute
 def count_training_frames(dt):
  command=term.command
  frame_counts[0]+=((command.abs()<.1).all(-1)).sum()
  frame_counts[1]+=((command==0).all(-1)).sum()
  frame_counts[2]+=command.shape[0]
  return original_reward_compute(dt)
 u.reward_manager.compute=count_training_frames
 manifest=dict(scheme=a.scheme,initialization='own_hip_A_scratch10k_resume',lineage='hip_A_random_scratch_0_to_10k_then_resume',base_checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),base_iteration=base_iteration,restored_actor_critic_optimizer=True,restored_learning_rate=initial_lr,additional_updates=a.iterations,total_updates=50000,seed=a.seed,num_envs=a.num_envs,num_steps_per_env=agent.num_steps_per_env,new_transitions=a.iterations*a.num_envs*agent.num_steps_per_env,total_transitions=50000*a.num_envs*agent.num_steps_per_env,hip_weights=hip_weights,hip_comfort_angle_rad=.25,hip_joint_order=HIP_NAMES,hip_orientation_gate=False,hip_command_gate=False,hip_warmup_iterations=0,actual_weight=-actual,target_weight=-target,wheel_baseline='previous_B',warmup_already_completed=True,reward_step_offset=base_iteration*agent.num_steps_per_env,initial_actual_weight=-actual,initial_target_weight=-target,command_mixture=[.2,.2,.6],native_training_pd=True,arm_feedforward=False,observation_noise=True,physical_gpu_uuid=os.environ.get('NVIDIA_VISIBLE_DEVICES'),stage_updates=list(range(10000,50001,1000)),evaluation_updates=[10000,20000,30000,40000,50000],geometry='no_camera_mounts_sdk_limits',parent_geometry='checkpoint-specific; verify parent manifest',evaluation_mode='synchronous child process at milestone; learner held in memory then continues',cumulative_frame_counts_scope='this additional 40000 update segment')
 save('manifest.json',manifest)
 evaluation_seconds=0.;evaluation_results={}
 def evaluate_stage(step):
  nonlocal evaluation_seconds
  before=time.time()
  save('status.json',dict(stage='evaluating',completed_updates=step,total_updates=50000,additional_completed_updates=step-base_iteration,updated_at=before))
  with (out/f'eval_{step:07d}.log').open('w') as log:
   process=subprocess.run([sys.executable,str(Path(__file__).with_name('eval_stage.py')),'--run',str(out),'--step',str(step),'--headless','--device',a.device],stdout=log,stderr=subprocess.STDOUT)
  evaluation_seconds+=time.time()-before
  status_path=out/'evaluation'/f'stage_{step:07d}'/'status.json'
  status=json.loads(status_path.read_text()) if status_path.exists() else {}
  ok=process.returncode==0 and status.get('stage')=='complete'
  evaluation_results[str(step)]=dict(exit_code=process.returncode,complete=ok,elapsed_seconds=time.time()-before,updated_at=time.time())
  save('evaluation_index.json',evaluation_results)
  # A failed eval is recorded without discarding this long training run; never auto-retry.
 evaluate_stage(10000)
 original_log=runner.logger.log;started=time.time();baseline_evaluation_seconds=evaluation_seconds
 def log_hook(*args,**kwargs):
  original_log(*args,**kwargs)
  completed=int(kwargs['it'])+1
  if completed%1000==0:
   old=runner.current_learning_iteration;runner.current_learning_iteration=completed
   stage=out/'stages';stage.mkdir(exist_ok=True)
   runner.save(str(stage/f'checkpoint_{completed:07d}.pt'),dict(base_iteration=base_iteration,additional_completed_updates=completed-base_iteration,manifest=manifest))
   runner.export_policy_to_jit(str(stage),f'actor_{completed:07d}.jit');runner.alg.train_mode();runner.current_learning_iteration=old
   save(f'stage_{completed:07d}.json',dict(completed_updates=completed,effective_iteration=completed,updated_at=time.time()))
   if completed in [20000,30000,40000,50000]:evaluate_stage(completed)
  if completed<=base_iteration+3 or completed%10==0:
   cmd=term.command;low=(cmd.abs()<.1).all(-1);actual_scale,target_scale=ramp(u.common_step_counter+u.training_step_offset,actual)
   info=dict(scheme=a.scheme,hip_weights=hip_weights,stage='training',updated_at=time.time(),completed_updates=completed,additional_completed_updates=completed-base_iteration,total_updates=50000,effective_iteration=completed,elapsed_seconds=time.time()-started,evaluation_seconds=evaluation_seconds-baseline_evaluation_seconds,training_seconds_excluding_evaluation=time.time()-started-(evaluation_seconds-baseline_evaluation_seconds),new_transitions=(completed-base_iteration)*a.num_envs*agent.num_steps_per_env,total_transitions=completed*a.num_envs*agent.num_steps_per_env,current_low_fraction=float(low.float().mean()),current_zero_fraction=float((cmd==0).all(-1).float().mean()),command_sample_counts=term.sample_counts.cpu().tolist(),cumulative_low_frame_fraction=float(frame_counts[0]/frame_counts[2].clamp(min=1)),cumulative_zero_frame_fraction=float(frame_counts[1]/frame_counts[2].clamp(min=1)),actual_weight=-actual*actual_scale,target_weight=-target*target_scale,learning_rate=runner.alg.learning_rate,losses={k:float(v) for k,v in kwargs['loss_dict'].items()})
   save('status.json',info)
   with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
   runner.logger.writer.add_scalar('LowSpeed/sample_low_fraction',float(term.sample_counts[:2].sum()/term.sample_counts.sum()),completed)
   runner.logger.writer.add_scalar('LowSpeed/current_low_fraction',info['current_low_fraction'],completed)
   runner.logger.writer.add_scalar('LowSpeed/actual_weight',info['actual_weight'],completed)
   runner.logger.writer.add_scalar('LowSpeed/target_weight',info['target_weight'],completed)
 runner.logger.log=log_hook
 save('status.json',dict(stage='loaded',updated_at=time.time(),base_iteration=base_iteration,completed_updates=base_iteration,initialization='own_hip_A_scratch10k_resume',hip_weights=hip_weights,actual_weight=-actual,target_weight=-target))
 runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
 save('status.json',dict(scheme=a.scheme,hip_weights=hip_weights,actual_weight=-actual,target_weight=-target,stage='training_complete',updated_at=time.time(),completed_updates=50000,additional_completed_updates=a.iterations,total_updates=50000,elapsed_seconds=time.time()-started,evaluation_seconds=evaluation_seconds-baseline_evaluation_seconds,training_seconds_excluding_evaluation=time.time()-started-(evaluation_seconds-baseline_evaluation_seconds),new_transitions=a.iterations*a.num_envs*agent.num_steps_per_env,total_transitions=50000*a.num_envs*agent.num_steps_per_env,evaluation_results=evaluation_results))
 env.close()
try:main()
except Exception as exc:
 save('status.json',dict(stage='failed',error=str(exc),traceback=traceback.format_exc(),updated_at=time.time()));traceback.print_exc();raise
finally:app.close()
