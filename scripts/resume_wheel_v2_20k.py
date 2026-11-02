"""Continue a completed WheelV2 stage; preserve PPO weights, optimizer and LR."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--total_iterations', type=int, default=20000)
p.add_argument('--expected_previous_iterations', type=int, default=5000)
p.add_argument('--num_envs', type=int, default=4096)
p.add_argument('--validate_only', action='store_true')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(a).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
import LeggedManip_Lab.tasks


def save(name, value):
    (a.output / name).write_text(json.dumps(value, indent=2) + '\n')


def same(x, y):
    if isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor) and torch.equal(x.cpu(), y.cpu())
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for key in x:
            same(x[key], y[key])
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for xx, yy in zip(x, y):
            same(xx, yy)
    else:
        assert x == y, (x, y)


def main():
    torch.set_num_threads(4)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    completed = int(saved['iter']) + 1  # runner stores last completed zero-based index
    assert completed == a.expected_previous_iterations, completed
    remaining = a.total_iterations - completed
    assert remaining > 0
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        assert key in saved
    cfg = parse_env_cfg('A2W-Z1-WBC-WheelV2', device=a.device, num_envs=a.num_envs)
    cfg.seed = 42
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.log_dir = str(a.output)
    assert cfg.actions.wheel_vel.scale == 5.
    assert cfg.actions.wheel_vel.clip == {'.*': (-100., 100.)}
    for command in (cfg.commands.ee_pose, cfg.commands.base_velocity):
        assert command.curriculum_enabled is False
    agent = load_cfg_from_registry('A2W-Z1-WBC-WheelV2', 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    agent.seed = 42
    agent.device = a.device
    agent.resume = True
    agent.load_run = a.checkpoint.parent.name
    agent.load_checkpoint = a.checkpoint.name
    agent.max_iterations = remaining
    agent.run_name = 'wheel_v2_seed42_continue_to20k'
    agent.logger = 'tensorboard'
    env = RslRlVecEnvWrapper(gym.make('A2W-Z1-WBC-WheelV2', cfg=cfg), clip_actions=agent.clip_actions)
    env.reset()
    # Native disabled-curriculum callbacks must select full command ranges.
    for name in ('ee_pose', 'base_velocity'):
        c = env.unwrapped.command_manager.get_term(name).cfg
        assert c.ranges.to_dict() == c.limit_ranges.to_dict()
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(a.output), device=a.device)
    runner.load(str(a.checkpoint), strict=True)
    assert runner.current_learning_iteration == saved['iter']
    restored = runner.alg.save()
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        same(saved[key], restored[key])
    # PPO.load restores optimizer.param_groups but not its separate adaptive-LR field.
    lrs = [group['lr'] for group in runner.alg.optimizer.param_groups]
    assert all(lr == lrs[0] for lr in lrs)
    runner.alg.learning_rate = lrs[0]
    runner.current_learning_iteration = completed
    dump_yaml(str(a.output / 'params/env.yaml'), cfg)
    dump_yaml(str(a.output / 'params/agent.yaml'), agent)
    manifest = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                    completed_iterations=completed, first_resume_index=completed,
                    additional_iterations=remaining, total_iterations=a.total_iterations,
                    final_checkpoint_index=a.total_iterations-1, loaded_actor_critic_optimizer_exact=True,
                    restored_learning_rate=lrs[0], num_envs=a.num_envs,
                    curriculum='disabled; verified full ranges after reset',
                    environment_state='reset; not a bitwise continuation of physics/RNG/rollout state',
                    validate_only=a.validate_only)
    save('resume_manifest.json', manifest)
    print('RESUME_VERIFIED', json.dumps(manifest), flush=True)
    if a.validate_only:
        assert torch.isfinite(env.get_observations()['policy']).all()
        save('status.json', dict(stage='VALIDATED', **manifest))
    else:
        save('status.json', dict(stage='TRAINING', **manifest))
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
        final = a.output / f'model_{a.total_iterations-1}.pt'
        assert final.is_file()
        assert torch.load(final, map_location='cpu', weights_only=False)['iter'] == a.total_iterations-1
        save('status.json', dict(stage='COMPLETE', final_checkpoint=str(final), **manifest))
    env.close()


try:
    main()
except Exception as exc:
    save('failure.json', {'error': repr(exc)})
    raise
finally:
    app.close()
