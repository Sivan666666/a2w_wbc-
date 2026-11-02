"""From-scratch A2W BPTT. PhysX collection, learned differentiable rollouts.

Run via reproduction/supermimic_runtime.sh; never connects to a real robot.
"""
import argparse
import json
import time
import os
import hashlib
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--task', default='A2W-Z1-WBC-WheelV2-Precision')
p.add_argument('--num_envs', type=int, default=4096)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--steps', type=int, default=120000, help='Physical control steps per environment, including warmup')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--resume', type=Path)
p.add_argument('--validate_only', action='store_true')
p.add_argument('--warmup', type=int, default=512)
p.add_argument('--buffer', type=int, default=128)
p.add_argument('--window', type=int, default=8)
p.add_argument('--wm_hidden', type=int, default=512)
p.add_argument('--wm_batch', type=int, default=256)
p.add_argument('--wm_updates', type=int, default=2)
p.add_argument('--imag_batch', type=int, default=256)
p.add_argument('--update_every', type=int, default=4)
p.add_argument('--horizon', type=int, default=16)
p.add_argument('--horizon_ramp', type=int, default=2048)
p.add_argument('--actor_lr', type=float, default=1e-4)
p.add_argument('--wm_lr', type=float, default=1e-3)
p.add_argument('--critic_lr', type=float, default=1e-3)
p.add_argument('--critic_epochs', type=int, default=2)
p.add_argument('--save_every', type=int, default=5000)
p.add_argument('--validate_every', type=int, default=128)
p.add_argument('--wm_gate', type=float, default=5., help='Holdout normalized loss ceiling before first actor update')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.num_envs >= 16 and args.buffer >= max(32,args.window)
args.output.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
import LeggedManip_Lab.tasks
from supermimic.core import Replay, TERMS, push_history
from supermimic.adapter import Adapter
from supermimic.learner import Learner
from supermimic.provenance import capture as capture_provenance
main_env = None


def main():
    global main_env
    capture_provenance(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.ee_pose.debug_vis = cfg.commands.base_velocity.debug_vis = False
    cfg.log_dir = str(args.output)
    dump_yaml(str(args.output/'env.yaml'), cfg)
    (args.output/'args.json').write_text(json.dumps(vars(args), default=str, indent=2))
    env = gym.make(args.task, cfg=cfg); main_env = env; e = env.unwrapped
    obs, _ = env.reset()
    bridge = Adapter(e)
    # Startup validation is free of policy optimization. These additional physics
    # steps are recorded in the budget and are also used in PPO validation.
    reports = []
    for t in range(8):
        obs, *_ = env.step(torch.zeros((args.num_envs,22), device=e.device))
        reports.append(bridge.validate())
    (args.output/'adapter_validation.json').write_text(json.dumps(reports, indent=2))
    print('ADAPTER_VALIDATION_PASS', reports[-1], flush=True)
    if args.validate_only:
        env.close(); return
    learner = Learner(bridge, args)
    replay = Replay(args.buffer, args.num_envs, e.device)
    writer = SummaryWriter(str(args.output/'tensorboard'))
    log = (args.output/'progress.jsonl').open('a', buffering=1)
    start = 0
    previous_wall = 0.
    validation_transitions = 8*args.num_envs
    if args.resume:
        old = torch.load(args.resume, map_location=e.device, weights_only=False)
        assert old['task'] == args.task and old['args']['num_envs'] == args.num_envs
        learner.load(old['learner']); start = old['step']
        previous_wall = old['wall_seconds']
        validation_transitions += old['env_transitions']-start*args.num_envs
        torch.set_rng_state(old['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in old['cuda_rng']])
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in learner.actor.state_dict().values())).hexdigest()
    metadata = dict(task=args.task, actor_obs=234, action_dim=22, state_dim=98,
                    initial_actor_sha256=initial_hash, from_scratch=args.resume is None,
                    privileged_context_dim=bridge.context().shape[-1], control_hz=50,
                    validation_physical_transitions=validation_transitions,
                    replay_resumed=False, heldout_env_ids=[replay.split,args.num_envs-1])
    (args.output/'manifest.json').write_text(json.dumps(metadata, indent=2))
    def save(step, milestone=False):
        data = dict(format='a2w-supermimic-v1', task=args.task, step=step, args=vars(args),
                    learner=learner.state_dict(), metadata=metadata, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), wall_seconds=previous_wall+time.monotonic()-started,
                    env_transitions=step*args.num_envs+validation_transitions,
                    resume_note='Replay/physics not serialized: fresh reset and mandatory world-model warmup on resume.')
        dest = args.output/(f'checkpoint_{step:07d}.pt' if milestone else 'latest.pt')
        tmp = dest.with_suffix('.tmp'); torch.save(data, tmp); os.replace(tmp,dest)
        # Export only the deterministic deployment interface, never the world model.
        if milestone:
            torch.jit.script(learner.actor).save(str(args.output/f'actor_{step:07d}.jit'))
    save(start, True)
    if args.resume is None:
        torch.manual_seed(args.seed+10000)
    obs, _ = env.reset()
    gate = False; wm_validation = {}; losses = {}; reward_acc = 0.; failures = 0; reward_count = 0
    for step in range(start+1,args.steps+1):
        with torch.no_grad():
            state = bridge.state().clone(); context = bridge.context()
            previous = e.action_manager.action.clone(); commands = bridge.commands()
            history = obs['policy'].clone(); age = e.episode_length_buf[:,None].float().clone()
            remaining = bridge.remaining()
            action = learner.action(history)
            assert torch.isfinite(action).all(), 'Nonfinite collection action'
            obs, reward, terminated, truncated, _ = env.step(action)
            capture = bridge.captured
            assert capture is not None and torch.isfinite(capture['target']).all()
            if step-start <= 32:
                # Verify actual Isaac Lab noisy history against our imagined
                # term-wise shift, not just a synthetic unit-test convention.
                valid = ~capture['done']; offset = 0
                for width in TERMS:
                    torch.testing.assert_close(obs['policy'][valid,offset:offset+2*width],
                                               history[valid,offset+width:offset+3*width],rtol=0,atol=0)
                    offset += 3*width
            if step-start in (32,64,128):
                # Nonzero random arm actions validate more than the initial pose.
                bridge.validate()
            replay.add(state=state, action=action, previous=previous, commands=commands, history=history,
                       context=context, age=age, remaining=remaining, **capture)
            # Normalization statistics only see training environments, never holdout.
            n = replay.split
            learner.model.update_stats(state[:n], action[:n], previous[:n], context[:n], age[:n], capture['target'][:n])
            reward_acc += float(reward.mean()); failures += int(terminated.sum()); reward_count += 1
        if replay.size >= args.window and step % args.update_every == 0:
            for _ in range(args.wm_updates):
                losses.update(learner.fit(replay.sample(args.wm_batch,args.window)))
            if step % args.validate_every == 0 and replay.size >= 32:
                wm_validation = learner.validate_model(replay.sample(args.wm_batch,32,validation=True))
                gate = wm_validation['wm_holdout_loss'] < args.wm_gate
            if step-start >= args.warmup and gate:
                horizon = min(args.horizon, 4+4*((step-start-args.warmup)//args.horizon_ramp))
                losses.update(learner.improve(replay.sample(args.imag_batch*2), horizon))
        if step-start > args.warmup+4096 and learner.actor_updates == 0:
            raise RuntimeError('World-model gate blocked all policy updates for 4096 steps after warmup')
        if step % 32 == 0:
            with torch.no_grad():
                metrics = bridge.metrics(bridge.proxy(capture['target'], action, previous, commands))
            row = dict(step=step, env_transitions=step*args.num_envs+validation_transitions,
                       wall_seconds=previous_wall+time.monotonic()-started, mean_reward=reward_acc/reward_count,
                       failure_resets=failures, actor_gate=gate, **losses, **wm_validation,
                       **{k:float(v) for k,v in metrics.items()})
            reward_acc=0.; failures=0; reward_count=0
            log.write(json.dumps(row)+'\n'); print(json.dumps(row), flush=True)
            for k,v in row.items():
                if k != 'step': writer.add_scalar(k,v,row['env_transitions'])
        if step % args.save_every == 0:
            save(step,True)
        elif step % 512 == 0:
            save(step)
    if learner.actor_updates == 0:
        raise RuntimeError('No actor updates: world-model validation gate never passed; not a successful training run')
    save(args.steps,True); save(args.steps)
    (args.output/'complete.json').write_text(json.dumps(dict(step=args.steps,actor_updates=learner.actor_updates)))
    writer.close(); log.close(); env.close()

try:
    main()
except Exception as exc:
    (args.output/'FAILED.txt').write_text(repr(exc))
    traceback.print_exc()
    raise
finally:
    if main_env is not None:
        main_env.close()
    app.close()
