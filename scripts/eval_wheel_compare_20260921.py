"""Read-only policy evaluation through the native Isaac Lab task and managers.

Metrics are captured after physics/reward computation, before reset and command
resampling. No extra observation/history update and no control modification.
"""
import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--num_envs', type=int, default=64)
p.add_argument('--seconds', type=float, default=200)
p.add_argument('--task', choices=['A2W-Z1-WBC-Play', 'A2W-Z1-WBC-WheelV2-Play'], required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)

def save(name, value):
    (a.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

save('status.json', {'stage': 'STARTING', 'seed': a.seed})
app = AppLauncher(a).app

import importlib.metadata
import subprocess
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_error_magnitude
import LeggedManip_Lab.tasks  # noqa: F401


def main():
    torch.set_num_threads(4)
    task = a.task
    cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs)
    cfg.seed = a.seed
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.log_dir = str(a.output)
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    agent.seed = a.seed
    agent.device = a.device
    dump_yaml(str(a.output / 'env.yaml'), cfg)
    dump_yaml(str(a.output / 'agent.yaml'), agent)
    raw = gym.make(task, cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    u = env.unwrapped
    assert u.num_envs == a.num_envs and u.action_manager.total_action_dim == 22
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=a.device)
    runner.load(str(a.checkpoint))
    policy = runner.get_inference_policy(device=u.device)
    obs = env.get_observations()
    assert obs['policy'].shape == (a.num_envs, 234), obs['policy'].shape
    robot = u.scene['robot']
    wheel_names = [f'{leg}_wheel_joint' for leg in ('FR', 'FL', 'RR', 'RL')]
    wheel_ids = robot.find_joints(wheel_names, preserve_order=True)[0]
    contact = u.scene['contact_forces']
    contact_ids = contact.find_bodies([f'{leg}_wheel' for leg in ('FR', 'FL', 'RR', 'RL')], preserve_order=True)[0]
    wheel_action = u.action_manager.get_term('wheel_vel')
    pose = u.command_manager.get_term('ee_pose')
    total = round(a.seconds / u.step_dt)
    assert math.isclose(total * u.step_dt, a.seconds)
    scalar_names = ['ee_native_m', 'ee_visual_world_m', 'orientation_rad',
                    'velocity_xy_m_s', 'yaw_rate_rad_s', 'tilt_deg', 'height_m']
    vector_sizes = {'ee_command': 7, 'velocity_command': 3,
                    'ee_position_w': 3, 'link_position_w': 3, 'link_quat_w': 4,
                    'root_position_w': 3, 'wheel_velocity': 4, 'wheel_target': 4,
                    'wheel_torque': 4, 'wheel_contact': 4, 'actual_velocity': 3}
    traces = {k: np.empty((total, a.num_envs), np.float32) for k in scalar_names}
    traces.update({k: np.empty((total, a.num_envs, n), np.float32)
                   for k, n in vector_sizes.items()})
    terms = list(u.termination_manager.active_terms)
    traces['termination_terms'] = np.empty((total, a.num_envs, len(terms)), bool)
    traces['terminated'] = np.empty((total, a.num_envs), bool)
    traces['timeout'] = np.empty((total, a.num_envs), bool)
    index = 0
    original_compute = u.reward_manager.compute

    def capture(dt):
        nonlocal index
        reward = original_compute(dt)
        ee = robot.data.body_pos_w[:, pose.body_idx]
        link = robot.data.body_pos_w[:, pose.link0_idx]
        q = robot.data.body_quat_w[:, pose.link0_idx]
        command = pose.command
        local = quat_apply_inverse(q, ee - link)
        mixed = torch.cat((local[:, :2] - command[:, :2], ee[:, 2:3] - command[:, 2:3]), -1)
        # Secondary metric: the exact target construction used by upstream debug visualization.
        delta = torch.zeros_like(ee)
        delta[:, :2] = command[:, :2]
        visual_goal = link + quat_apply(q, delta)
        visual_goal[:, 2] = command[:, 2]
        velocity = u.command_manager.get_command('base_velocity')
        values = {
            'ee_native_m': mixed.norm(dim=-1),
            'ee_visual_world_m': (ee - visual_goal).norm(dim=-1),
            'orientation_rad': quat_error_magnitude(
                robot.data.body_quat_w[:, pose.body_idx], quat_mul(q, command[:, 3:])),
            'velocity_xy_m_s': (robot.data.root_lin_vel_b[:, :2] - velocity[:, :2]).norm(dim=-1),
            'yaw_rate_rad_s': (robot.data.root_ang_vel_b[:, 2] - velocity[:, 2]).abs(),
            'tilt_deg': torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)) * 180 / math.pi,
            'height_m': robot.data.root_pos_w[:, 2] - u.scene.env_origins[:, 2],
            'ee_command': command, 'velocity_command': velocity,
            'ee_position_w': ee, 'link_position_w': link, 'link_quat_w': q,
            'root_position_w': robot.data.root_pos_w,
            'wheel_velocity': robot.data.joint_vel[:, wheel_ids],
            'wheel_target': wheel_action.processed_actions,
            'wheel_torque': robot.data.applied_torque[:, wheel_ids],
            'wheel_contact': (contact.data.net_forces_w[:, contact_ids].norm(dim=-1) > 1.).float(),
            'actual_velocity': torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), -1),
        }
        for key, value in values.items():
            array = value.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise RuntimeError(f'Nonfinite {key} at step {index}')
            traces[key][index] = array
        traces['terminated'][index] = u.termination_manager.terminated.cpu().numpy()
        traces['timeout'][index] = u.termination_manager.time_outs.cpu().numpy()
        for j, name in enumerate(terms):
            traces['termination_terms'][index, :, j] = u.termination_manager.get_term(name).cpu().numpy()
        index += 1
        return reward

    u.reward_manager.compute = capture
    root = Path.cwd()
    files = [Path(__file__), root / 'scripts/rsl_rl/play.py']
    files += sorted((root / 'source/LeggedManip_Lab/LeggedManip_Lab').rglob('*.py'))
    source_hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=root)
    (a.output / 'workspace.diff').write_bytes(diff)
    save('manifest.json', {
        'task': task, 'checkpoint': str(a.checkpoint),
        'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'source_hashes': source_hashes, 'workspace_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'seed': a.seed, 'num_envs': a.num_envs, 'seconds_per_env': a.seconds,
        'step_dt': u.step_dt, 'steps': total, 'termination_terms': terms,
        'protocol': 'Native PLAY config: policy observation noise off; native startup/reset randomization and actuator delays retained. Native command distributions and termination rules unchanged. Metrics after physics, before reset/resampling. All frames; no warmup removal.',
        'ee_native_definition': 'norm([link0-local x/y error, world z error]); NOT a pure world Euclidean norm when tilted.',
        'ee_visual_world_definition': 'World Euclidean error to upstream visualization target: link0 + R_link0*[command_x,command_y,0], then overwrite world z.',
        'velocity_definition': 'Per-frame body-frame XY Euclidean error and absolute yaw-rate error; NOT accumulated TensorBoard command metrics.',
        'wheel_definition': 'FR FL RR RL order. Target is policy processed velocity target BEFORE actuator delay. Contact is net force norm >1N. Rotation/contact statistics alone do not establish no-slip rolling.',
    })
    with torch.inference_mode():
        for i in range(total):
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise RuntimeError('Nonfinite actions')
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            if (i + 1) % 500 == 0:
                state = dict(stage='EVALUATING', seed=a.seed, steps=i+1, total_steps=total,
                             mean_native_ee_cm=float(traces['ee_native_m'][:i+1].mean()*100))
                save('status.json', state)
                print('PROGRESS', json.dumps(state), flush=True)
    assert index == total
    np.savez_compressed(a.output / 'trace.npz', **traces)
    summary = dict(seed=a.seed, num_envs=a.num_envs, seconds_per_env=a.seconds,
                   samples=total*a.num_envs, checkpoint=str(a.checkpoint),
                   non_timeout_resets=int((traces['terminated'] & ~traces['timeout']).sum()),
                   termination_events=int(traces['terminated'].sum()),
                   timeout_events=int(traces['timeout'].sum()),
                   total_resets=int((traces['terminated'] | traces['timeout']).sum()),
                   termination_counts={k: int(traces['termination_terms'][:, :, j].sum()) for j,k in enumerate(terms)},
                   tilt_over_60_fraction=float((traces['tilt_deg'] > 60).mean()))
    for k in scalar_names:
        values = traces[k].astype(np.float64)
        summary[k] = dict(mean=float(values.mean()), rmse=float(np.sqrt(np.mean(values**2))),
                          p95=float(np.quantile(values, .95)), max=float(values.max()))
    moving = np.linalg.norm(traces['velocity_command'][:, :, :2], axis=-1) > .3
    summary['wheels'] = {}
    for name, mask in [('all', np.ones_like(moving)), ('moving_command_xy_over_0p3', moving)]:
        v = traces['wheel_velocity'][mask]
        target = traces['wheel_target'][mask]
        c = traces['wheel_contact'][mask]
        summary['wheels'][name] = dict(samples=int(mask.sum()),
            mean_abs_velocity=float(np.abs(v).mean()), mean_abs_target=float(np.abs(target).mean()),
            target_mae=float(np.abs(v-target).mean()), contact_fraction=float(c.mean()),
            contact_and_abs_speed_gt1_fraction=float(((c>0) & (np.abs(v)>1)).mean()),
            mean_abs_torque=float(np.abs(traces['wheel_torque'][mask]).mean()))
    save('summary.json', summary)
    print('SUMMARY', json.dumps(summary), flush=True)
    env.close()
    save('status.json', dict(stage='COMPLETE', seed=a.seed, steps=total))


try:
    main()
except BaseException:
    save('status.json', dict(stage='FAILED', traceback=traceback.format_exc()))
    raise
finally:
    app.close()
