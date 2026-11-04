"""From-scratch Go2-aligned learner on the unchanged A2W WheelV2 Precision task."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=4096)
p.add_argument('--iterations',type=int,default=40000)
p.add_argument('--imag_chunk',type=int,default=1024,help='Execution memory chunk; effective batch remains8192')
p.add_argument('--smoke',action='store_true',help='Validation run, not a production result; budget/env may be smaller')
p.add_argument('--eager',action='store_true',help='Diagnostic only; production uses upstream compile=True')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
app=AppLauncher(args).app
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
import LeggedManip_Lab.tasks
from go2_aligned.upstream import TrainCfg
from go2_aligned.bridge import Bridge,pose_delta
from go2_aligned.replay import Replay
from go2_aligned.learner import Learner
from supermimic.provenance import capture
main_env=None


def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,default=str));os.replace(tmp,path)


def main():
    global main_env
    if not args.smoke:
        assert args.num_envs==4096 and args.iterations==40000 and not args.eager
    c=TrainCfg(task='A2W-Z1-WBC-WheelV2-Precision',num_envs=args.num_envs,
               iterations=args.iterations,compile=not args.eager)
    torch.set_num_threads(4);torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    capture(args.output)
    source=Path(__file__).parent
    hashes={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),*sorted((source/'go2_aligned').glob('*.py'))]}
    write_json(args.output/'config.json',dataclasses.asdict(c))
    write_json(args.output/'manifest.json',dict(format='a2w-go2-aligned-v1',from_scratch=True,
        upstream='Zhengtianjin/go2-supermimic@d9eee9b3f248e5300e636c433255db2c1c3e086a',
        source_sha256=hashes,smoke=args.smoke,execution_chunk=args.imag_chunk,effective_imag_batch=c.imag_batch,
        training_hardware='PS1 physical GPU3',reward_task='WheelV2 Precision; not Go2 chase/dodge',
        seed=c.seed,reference_ppo_seed=42,reference_ppo_hardware='A100',
        differences=['22 actions/234 history inputs;3-frame A2W history','86 physical+limb state,766 context;77 flat scan',
          'A2W EE rewards/commands/assets/50Hz control/200Hz physics retained',
          'A2W1rad tilt and geometric base-ground soft contact;no learned termination',
          'Go2 analytic PD/acc reward surrogate adapted to delayed wheel/leg/arm control',
          'replay pre-reset target stored;terminal transition masked as upstream',
          'fixed-command interior seeds;no chase/dodge/terrain curriculum',
          'execution gradient accumulation;unchanged full Actor batch and critic minibatches'],
        no_old_checkpoint_loaded=True,not_a_paper_reproduction=True))
    ec=parse_env_cfg(c.task,device=args.device,num_envs=c.num_envs);ec.seed=c.seed
    ec.scene.terrain.visual_material=None;ec.scene.sky_light.spawn.texture_file=None
    ec.commands.ee_pose.debug_vis=ec.commands.base_velocity.debug_vis=False
    ec.log_dir=str(args.output)
    dump_yaml(str(args.output/'env.yaml'),ec)
    env=gym.make(c.task,cfg=ec);main_env=env;e=env.unwrapped
    obs,_=env.reset();b=Bridge(e)
    for _ in range(8):obs,*_=env.step(torch.zeros((c.num_envs,22),device=e.device))
    write_json(args.output/'adapter_validation.json',b.validate_aligned())
    print('GO2_ALIGNED_ADAPTER_PASS',flush=True)
    learner=Learner(b,c,args.imag_chunk)
    replay=Replay(c,b.static,b.context().shape[1]-b.static_dim)
    writer=SummaryWriter(str(args.output/'tensorboard'))
    progress=(args.output/'progress.jsonl').open('a',buffering=1)
    physical_steps=8;actor_steps=0;imag_transitions=0;failures=0;reward_sum=0.;reward_count=0
    ou=torch.zeros((c.num_envs,22),device=e.device)
    explore=torch.arange(c.num_envs,device=e.device)<int(c.explore_frac*c.num_envs)
    falls=[];collapse_strikes=0;last_safe=None

    def save(it):
        path=args.output/f'checkpoint_{it:07d}.pt'
        data=dict(format='a2w-go2-aligned-v1',iteration=it,step=it,config=dataclasses.asdict(c),
                  learner=learner.state_dict(),env_transitions=physical_steps*c.num_envs,
                  physical_steps_per_env=physical_steps,wall_seconds=time.monotonic()-started,
                  imagined_transitions=imag_transitions,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                  note='No replay/PhysX serialization;do not claim exact resume. Actor includes RunningNorm.')
        tmp=path.with_suffix('.tmp');torch.save(data,tmp);os.replace(tmp,path)
        actor_path=args.output/f'actor_{it:07d}.jit'
        torch.jit.script(learner.actor).save(str(actor_path))
        exported=torch.jit.load(str(actor_path),map_location=e.device)
        with torch.no_grad():torch.testing.assert_close(exported(obs['policy']),learner.actor(obs['policy']),rtol=1e-5,atol=1e-6)
        write_json(args.output/'latest_checkpoint.json',dict(path=str(path),iteration=it))
        return path

    @torch.no_grad()
    def collect():
        nonlocal obs,ou,physical_steps,failures,reward_sum,reward_count
        s=b.aligned_state();pose=b.pose();ctx=b.context();previous=e.action_manager.action.clone()
        commands=b.commands();history=obs['policy'].clone();age=e.episode_length_buf[:,None].float().clone()
        remaining=b.remaining()
        action=learner.actor(history)
        ou=ou*(1-c.explore_theta*e.step_dt)+c.explore_noise*(2*c.explore_theta*e.step_dt)**.5*torch.randn_like(action)
        action=action+torch.where(explore[:,None],ou,c.action_noise*torch.randn_like(action))
        assert torch.isfinite(action).all()
        obs,r,term,trunc,extras=env.step(action)
        cap=b.captured;done=term|trunc
        target=cap['aligned'];dp=pose_delta(pose,cap['pose'])
        assert torch.isfinite(target).all()
        replay.add(s=s,target=target,a=action,previous=previous,history=history,commands=commands,
                   dynamic=ctx[:,b.static_dim:],dpose=dp,remaining=remaining,age=age,done=done[:,None])
        keep=~done
        if keep.any():learner.wm.update_stats(s[keep],action[keep],previous[keep],ctx[keep],
                                             torch.zeros((int(keep.sum()),77),device=e.device),target[keep],dp[keep])
        learner.actor.norm.update(history)
        ou=torch.where(done[:,None],torch.zeros_like(ou),ou)
        physical_steps+=1;failures+=int(term.sum());reward_sum+=float(r.mean());reward_count+=1
        logs=extras.get('log',{})
        # Go2 guard consumes episode-termination log averages, NOT raw reset counts.
        if 'Episode_Termination/bad_orientation' in logs:
            falls.append(float(logs['Episode_Termination/bad_orientation']))
        return b.metrics(b.proxy(cap['target'],action,previous,commands))

    last_safe=save(0)
    while replay.size<=c.wm_window+1:collect()
    print('GO2_ALIGNED_PREFILL_PASS',flush=True)
    for it in range(1,c.iterations+1):
        lr,actor_lr=learner.schedule(it)
        for _ in range(c.collect_steps):metrics=collect()
        fit=[]
        for _ in range(c.wm_updates):fit.append(learner.fit(replay.wm_batch())[0])
        stats=dict(wm_loss=sum(fit)/len(fit))
        if it>c.warmup_iterations:
            seed=replay.seeds(c.imag_batch,(c.horizon+1)*e.step_dt)
            stats.update(learner.improve(seed));imag_transitions+=c.imag_batch*c.horizon
            if it==c.warmup_iterations+1:print('GO2_ALIGNED_FIRST_ACTOR_UPDATE',json.dumps(stats),flush=True)
        if it%c.log_interval==0 or it==c.iterations or it==c.warmup_iterations+1:
            row=dict(iteration=it,step=physical_steps,env_transitions=physical_steps*c.num_envs,
                     wall_seconds=time.monotonic()-started,imagined_transitions=imag_transitions,
                     mean_reward=reward_sum/reward_count,failure_resets=failures,lr=lr,actor_lr=actor_lr,
                     actor_updates=learner.actor_updates,archive_size=replay.archive_size,
                     gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(),
                     **{k:float(v) for k,v in metrics.items()})
            row.update(stats)
            progress.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            for k,v in row.items():writer.add_scalar(k,v,physical_steps*c.num_envs)
            reward_sum=0.;reward_count=0;failures=0
            falls_mean=sum(falls)/len(falls) if falls else 0.;falls.clear()
            collapse_strikes=collapse_strikes+1 if falls_mean>c.collapse_falls else 0
            if collapse_strikes>=2 and last_safe is not None:
                learner.rollback(last_safe,it);collapse_strikes=0
                event=dict(iteration=it,falls=falls_mean,checkpoint=str(last_safe),lr_scale=learner.lr_scale,
                           action='upstream collapse guard: actor/critic rollback;WM retained')
                with (args.output/'rollback_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                print('COLLAPSE_GUARD',json.dumps(event),flush=True)
        if (it%c.save_interval==0 or it==c.iterations) and collapse_strikes==0:last_safe=save(it)
    final=save(c.iterations)
    assert learner.actor_updates>0 or c.iterations<=c.warmup_iterations
    write_json(args.output/'complete.json',dict(iteration=c.iterations,actor_updates=learner.actor_updates,
              checkpoint=str(final),env_transitions=physical_steps*c.num_envs))
    writer.close();progress.close();env.close()


try:main()
except Exception as exc:
    write_json(args.output/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    traceback.print_exc();raise
finally:
    if main_env is not None:main_env.close()
    app.close()
