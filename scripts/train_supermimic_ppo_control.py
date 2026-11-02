"""Unmodified RSL-RL PPO algorithm, paired task and random initialization.

The initialization file must be BPTT step ZERO, never a pretrained policy.
"""
import argparse
import json
import time
import hashlib
import importlib.metadata
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--initial',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=4096)
p.add_argument('--iterations',type=int,default=5000)
p.add_argument('--seed',type=int,default=42)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
started=time.monotonic(); app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks
from supermimic.adapter import Adapter
from supermimic.provenance import capture
main_env=None


def main():
    global main_env
    capture(a.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    initial=torch.load(a.initial,map_location=a.device,weights_only=False)
    assert initial['step']==0 and initial['metadata']['from_scratch']
    assert initial['args']['seed']==a.seed
    task=initial['task']
    cfg=parse_env_cfg(task,device=a.device,num_envs=a.num_envs);cfg.seed=a.seed
    cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
    cfg.commands.ee_pose.debug_vis=cfg.commands.base_velocity.debug_vis=False
    cfg.log_dir=str(a.output)
    dump_yaml(str(a.output/'env.yaml'),cfg)
    env=gym.make(task,cfg=cfg);main_env=env;env.reset();bridge=Adapter(env.unwrapped)
    reports=[]
    for _ in range(8):
        env.step(torch.zeros((a.num_envs,22),device=a.device));reports.append(bridge.validate())
    (a.output/'adapter_validation.json').write_text(json.dumps(reports,indent=2))
    # The PPO rollout does not need any WM state capture; remove its cost.
    env.unwrapped.reward_manager.compute=bridge.original_reward_compute
    env.reset()
    agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
    agent.seed=a.seed;agent.logger='tensorboard';agent.save_interval=250
    agent.resume=False
    dump_yaml(str(a.output/'agent.yaml'),agent)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=str(a.output),device=a.device)
    policy=runner.alg.get_policy()
    assert not policy.obs_normalization
    policy.mlp.load_state_dict(initial['learner']['actor'],strict=True)
    runner.alg.critic.mlp.load_state_dict(initial['learner']['critic'],strict=True)
    policy.distribution.log_std_param.data.copy_(initial['learner']['log_std'])
    digest=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in policy.mlp.state_dict().values())).hexdigest()
    assert digest==initial['metadata']['initial_actor_sha256']
    manifest=dict(initial_actor_sha256=digest,from_scratch=True,task=task,
                  iterations=a.iterations,num_envs=a.num_envs,seed=a.seed,
                  env_transitions_budget=(a.iterations*24+8)*a.num_envs,
                  initial_source=str(a.initial),wall_started=time.time())
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    original_save=runner.save
    def save(path,infos=None):
        original_save(path,infos)
        step=(runner.current_learning_iteration+1)*24
        torch.jit.script(policy.mlp).save(str(a.output/f'actor_{step:07d}.jit'))
        with (a.output/'checkpoint_index.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,env_transitions=(step+8)*a.num_envs,
                                        wall_seconds=time.monotonic()-started,checkpoint=path))+'\n')
    runner.save=save
    torch.manual_seed(a.seed+10000)
    env.reset()
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    torch.jit.script(policy.mlp).save(str(a.output/'actor_final.jit'))
    (a.output/'complete.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-started,**manifest)))
    env.close()

try: main()
except Exception as exc:
    (a.output/'FAILED.txt').write_text(repr(exc));traceback.print_exc();raise
finally:
    if main_env is not None: main_env.close()
    app.close()
