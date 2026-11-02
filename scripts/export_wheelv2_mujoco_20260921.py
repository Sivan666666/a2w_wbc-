"""Export with the native RSL-RL runner; save simulator states for bridge parity checks."""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(a).app
import gymnasium as gym
import importlib.metadata
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks

try:
    cfg = parse_env_cfg('A2W-Z1-WBC-WheelV2-Play', device=a.device, num_envs=4)
    cfg.seed = 20260921
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    agent = load_cfg_from_registry('A2W-Z1-WBC-WheelV2-Play', 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    agent.device = a.device
    env = RslRlVecEnvWrapper(gym.make('A2W-Z1-WBC-WheelV2-Play', cfg=cfg), clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=a.device)
    runner.load(str(a.checkpoint))
    policy = runner.get_inference_policy(device=a.device)
    runner.export_policy_to_jit(path=str(a.output), filename='policy.pt')
    runner.export_policy_to_onnx(path=str(a.output), filename='policy.onnx')
    jit = torch.jit.load(str(a.output/'policy.pt'), map_location=a.device).eval()
    u = env.unwrapped
    robot = u.scene['robot']
    names = [f'{leg}_{j}_joint' for leg in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]
    names += [f'joint{i}' for i in range(1,7)] + [f'{leg}_wheel_joint' for leg in ('FR','FL','RR','RL')]
    ids = robot.find_joints(names, preserve_order=True)[0]
    body_names = ['base','link0','end_effector']
    bids = robot.find_bodies(body_names, preserve_order=True)[0]
    data = {k: [] for k in ['obs','action','root_pose','root_lin_vel_b','root_ang_vel_b','q','dq','body_pose','ee_command','vel_command']}
    obs = env.get_observations()
    max_error = 0.
    with torch.inference_mode():
        for i in range(100):
            action = policy(obs)
            error = float((jit(obs['policy'])-action).abs().max())
            max_error = max(max_error, error)
            torch.testing.assert_close(jit(obs['policy']), action, rtol=1e-5, atol=1e-5)
            values = dict(obs=obs['policy'], action=action,
                root_pose=robot.data.root_pose_w, root_lin_vel_b=robot.data.root_lin_vel_b,
                root_ang_vel_b=robot.data.root_ang_vel_b, q=robot.data.joint_pos[:,ids],
                dq=robot.data.joint_vel[:,ids], body_pose=robot.data.body_state_w[:,bids,:7],
                ee_command=u.command_manager.get_command('ee_pose'),
                vel_command=u.command_manager.get_command('base_velocity'))
            for k, v in values.items():
                data[k].append(v.cpu().numpy().copy())
            obs, _, dones, _ = env.step(action)
            policy.reset(dones)
    np.savez_compressed(a.output/'native_parity_states.npz', **{k:np.stack(v) for k,v in data.items()})
    dump_yaml(str(a.output/'env.yaml'), cfg)
    dump_yaml(str(a.output/'agent.yaml'), agent)
    manifest = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        jit_sha256=hashlib.sha256((a.output/'policy.pt').read_bytes()).hexdigest(),
        exported_with='native OnPolicyRunner.export_policy_to_jit/onnx, includes actor normalizer',
        joint_names=names, body_names=body_names, jit_parity_max_abs_error=max_error,
        samples=400, obs_dim=234, action_dim=22, step_dt=u.step_dt, physics_dt=u.physics_dt)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('EXPORT_PASS', json.dumps(manifest), flush=True)
    env.close()
finally:
    app.close()
