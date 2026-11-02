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
p.add_argument('--seconds', type=float, default=48)
p.add_argument('--variant', choices=['baseline', 'precision'], required=True)
p.add_argument('--replay_trace', type=Path)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
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
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_error_magnitude
import LeggedManip_Lab.tasks  # noqa: F401
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

PHASES = [
    ('FORWARD', (1., 0., 0.)),
    ('SIDE +Y', (0., .2, 0.)),
    ('SIDE -Y', (0., -.2, 0.)),
    ('TURN +YAW', (0., 0., .5)),
    ('TURN -YAW', (0., 0., -.5)),
    ('FORWARD + SIDE + TURN', (.6, .15, .35)),
]
PHASE_SECONDS = 8.
EXPECTED_HASHES = {'baseline': 'f8c576a5a8a297d3c43e3b350e9d58af630e217722ee818d0b7930ef1c82d633',
                  'precision': '378e5fd30611d5de9a1c61db56bfbfdc2d054128a80a2f1b04c7ece94ec15fee'}
LABEL = 'WheelV2 50K' if a.variant == 'baseline' else 'Precision 50K'
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_wbc import UniformPoseWBCCommand

class ReplayPoseCommand(UniformPoseWBCCommand):
    """Replay baseline's input commands before observation updates, without changing policy/control."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        values = np.load(a.replay_trace)['ee_command'][:, 0]
        self.replay = torch.tensor(values, device=self.device)
    def _update_command(self):
        idx = min(self._env.common_step_counter, len(self.replay)-1)
        self.pose_command[:] = self.replay[idx]
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._update_command()



class ScheduledVelocityCommand(UniformVelocityCommand):
    """Inject schedule during native command updates, before observation/history updates."""
    def _update_command(self):
        super()._update_command()
        t = self._env.common_step_counter * self._env.step_dt
        phase = min(int((t + 1e-8) / PHASE_SECONDS), len(PHASES)-1)
        self.vel_command_b[:] = torch.tensor(PHASES[phase][1], device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._update_command()


def main():
    torch.set_num_threads(4)
    task = 'A2W-Z1-WBC-WheelV2-Play'
    cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs)
    assert a.num_envs == 1
    assert a.seconds == PHASE_SECONDS * len(PHASES)
    assert cfg.actions.wheel_vel.scale == 5.
    assert cfg.actions.wheel_vel.clip == {'.*': (-100., 100.)}
    assert cfg.scene.robot.actuators['wheels'].damping == .6
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == EXPECTED_HASHES[a.variant]
    cfg.seed = a.seed
    if a.replay_trace:
        cfg.commands.ee_pose.class_type = ReplayPoseCommand
    for r in (cfg.commands.ee_pose.ranges, cfg.commands.ee_pose.limit_ranges):
        r.pos_x = (0.55, 0.85)
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.class_type = ScheduledVelocityCommand
    # Keep trained ranges unchanged; every scheduled command must be in-domain.
    limits = cfg.commands.base_velocity.limit_ranges
    for _, cmd in PHASES:
        for val, attr in zip(cmd, ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')):
            lo, hi = getattr(limits, attr)
            assert lo <= val <= hi
    # Continuous demonstration: avoid the native 20s timeouts; retain failure terminations.
    cfg.episode_length_s = 60.
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.commands.base_velocity.heading_command = False
    cfg.viewer.resolution = (1280, 720)
    cfg.log_dir = str(a.output)
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    agent.seed = a.seed
    agent.device = a.device
    dump_yaml(str(a.output / 'env.yaml'), cfg)
    dump_yaml(str(a.output / 'agent.yaml'), agent)
    raw = gym.make(task, cfg=cfg, render_mode='rgb_array')
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    u = env.unwrapped
    assert u.num_envs == a.num_envs and u.action_manager.total_action_dim == 22
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=a.device)
    runner.load(str(a.checkpoint))
    policy = runner.get_inference_policy(device=u.device)
    obs = env.get_observations()
    assert obs['policy'].shape == (a.num_envs, 234), obs['policy'].shape
    robot = u.scene['robot']
    pose = u.command_manager.get_term('ee_pose')
    # Visual-only spheres: no rigid-body or collision properties, no effect on control.
    markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path='/Visuals/EEPositionSpheres',
        markers={
            'target': sim_utils.SphereCfg(radius=.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., .9, 1.))),
            'actual': sim_utils.SphereCfg(radius=.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1., .4, .02))),
        },
    ))
    total = round(a.seconds / u.step_dt)
    assert math.isclose(total * u.step_dt, a.seconds)
    scalar_names = ['ee_native_m', 'ee_visual_world_m', 'orientation_rad',
                    'velocity_xy_m_s', 'yaw_rate_rad_s', 'tilt_deg', 'height_m']
    vector_sizes = {'ee_command': 7, 'velocity_command': 3,
                    'ee_position_w': 3, 'link_position_w': 3, 'link_quat_w': 4,
                    'root_position_w': 3, 'actual_velocity': 3}
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
        if a.replay_trace:
            assert torch.equal(command[0], pose.replay[index]), f'EE command mismatch at {index}'
        local = quat_apply_inverse(q, ee - link)
        mixed = torch.cat((local[:, :2] - command[:, :2], ee[:, 2:3] - command[:, 2:3]), -1)
        # Secondary metric: the exact target construction used by upstream debug visualization.
        delta = torch.zeros_like(ee)
        delta[:, :2] = command[:, :2]
        visual_goal = link + quat_apply(q, delta)
        visual_goal[:, 2] = command[:, 2]
        velocity = u.command_manager.get_command('base_velocity')
        phase_index = min(int((index*u.step_dt + 1e-8)/PHASE_SECONDS), len(PHASES)-1)
        assert torch.allclose(velocity, velocity.new_tensor(PHASES[phase_index][1]).expand_as(velocity), atol=1e-6)
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
            'actual_velocity': torch.cat((robot.data.root_lin_vel_b[:,:2],robot.data.root_ang_vel_b[:,2:3]),-1),
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
        'variant': a.variant, 'replay_trace': str(a.replay_trace) if a.replay_trace else None,
        'ee_x_range': [0.55,0.85],
        'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'source_hashes': source_hashes, 'workspace_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'seed': a.seed, 'num_envs': a.num_envs, 'seconds_per_env': a.seconds,
        'step_dt': u.step_dt, 'steps': total, 'termination_terms': terms,
        'protocol': 'Six 8s phases of body-frame velocity commands. Baseline samples native EE goals; precision replays identical 7D commands frame by frame. PLAY randomization, delays and failure terminations retained. Timeout extended to 60s for continuous 48s video. NOT the 64-env benchmark.',
        'command_schedule': [{'start_s':j*PHASE_SECONDS, 'end_s':(j+1)*PHASE_SECONDS, 'name':label, 'command':cmd} for j,(label,cmd) in enumerate(PHASES)],
        'ee_native_definition': 'norm([link0-local x/y error, world z error]); NOT a pure world Euclidean norm when tilted.',
        'ee_visual_world_definition': 'World Euclidean error to upstream visualization target: link0 + R_link0*[command_x,command_y,0], then overwrite world z.',
        'velocity_definition': 'Per-frame body-frame XY Euclidean error and absolute yaw-rate error; NOT accumulated TensorBoard command metrics.',
    })
    writer = imageio.get_writer(str(a.output/'play.mp4'), fps=25, codec='libx264', quality=8, macro_block_size=1)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    cyan, orange = '#00e5ff', '#ff9b36'
    marker_positions = []
    frames = 0
    with torch.inference_mode():
        for i in range(total):
            phase = min(int((i*u.step_dt + 1e-8) / PHASE_SECONDS), len(PHASES)-1)
            expected = torch.tensor(PHASES[phase][1], device=u.device)
            assert torch.allclose(u.command_manager.get_command('base_velocity')[0], expected), (i, phase)
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise RuntimeError('Nonfinite actions')
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            if i % 2 == 0:
                base = robot.data.root_pos_w[0].cpu().numpy()
                u.sim.set_camera_view(eye=base+np.array([2.2,-3.0,1.6]), target=base+np.array([.35,0,.15]))
                # Match current rendered state (including post-reset state), not stale pre-reset telemetry.
                ee_now = robot.data.body_pos_w[0,pose.body_idx]
                link_now = robot.data.body_pos_w[0,pose.link0_idx]
                q_now = robot.data.body_quat_w[0,pose.link0_idx]
                delta = torch.zeros((1,3),device=u.device)
                delta[0,:2] = pose.command[0,:2]
                goal_now = link_now + quat_apply(q_now.unsqueeze(0),delta)[0]
                goal_now[2] = pose.command[0,2]
                positions = torch.stack((goal_now, ee_now))
                markers.visualize(translations=positions, marker_indices=[0, 1])
                marker_positions.append(positions.cpu().numpy())
                rgb = raw.render()
                im = Image.fromarray(rgb)
                draw = ImageDraw.Draw(im)
                draw.rectangle((0,0,1280,118), fill=(20,25,34))
                v = traces['actual_velocity'][i,0]
                draw.text((15,10), f'{LABEL} | {phase+1}/6 {PHASES[phase][0]} | t={(i+1)*u.step_dt:.2f}s',font=font,fill='white')
                cmd = traces['velocity_command'][i,0]
                draw.text((15,44), f'CMD vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} vyaw={cmd[2]:+.2f} | ACT {v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}',font=font,fill='white')
                fail = int((traces['terminated'][:i+1] & ~traces['timeout'][:i+1]).sum())
                visual_error = float((goal_now-ee_now).norm().item()*100)
                local_now = quat_apply_inverse(q_now.unsqueeze(0), (ee_now-link_now).unsqueeze(0))[0]
                err_now = torch.cat((local_now[:2]-pose.command[0,:2], (ee_now[2]-pose.command[0,2]).reshape(1)))
                native_error = float(err_now.norm().item()*100)
                draw.text((15,78), f'EE native={native_error:.1f} cm | world target={visual_error:.1f} cm | resets={fail}',font=font,fill='white')
                draw.rectangle((0,670,1280,720),fill=(20,25,34))
                draw.text((15,681),'CYAN SPHERE: TARGET',font=small,fill=cyan)
                draw.text((350,681),'ORANGE SPHERE: ACTUAL EE',font=small,fill=orange)
                draw.text((810,681),'BODY FRAME | m/s, m/s, rad/s',font=small,fill='white')
                if bool(dones[0]):
                    draw.text((15,130),'EPISODE RESET (native timeout / termination)',font=small,fill='yellow')
                writer.append_data(np.asarray(im))
                if frames in [int((j*PHASE_SECONDS+4)*25) for j in range(len(PHASES))]: im.save(a.output/f'preview_{frames:04d}.jpg')
                frames += 1
            if (i + 1) % 500 == 0:
                state = dict(stage='EVALUATING', seed=a.seed, steps=i+1, total_steps=total,
                             mean_native_ee_cm=float(traces['ee_native_m'][:i+1].mean()*100))
                save('status.json', state)
                print('PROGRESS', json.dumps(state), flush=True)
    assert index == total
    writer.close()
    np.save(a.output/'marker_positions_world.npy',np.asarray(marker_positions))
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
    summary['variant'] = a.variant
    summary['checkpoint_sha256'] = EXPECTED_HASHES[a.variant]
    summary['video_frames'] = frames
    summary['video_fps'] = 25
    save('summary.json', summary)
    phases = []
    for j, (label, cmd) in enumerate(PHASES):
        start = round(j*PHASE_SECONDS/u.step_dt)
        end = round((j+1)*PHASE_SECONDS/u.step_dt)
        trim = start + round(2/u.step_dt)
        actual = traces['actual_velocity'][start:end,0]
        assert np.allclose(traces['velocity_command'][start:end,0], cmd)
        phases.append(dict(name=label, command=cmd, start_s=j*PHASE_SECONDS,
                           mean_actual=actual.mean(axis=0).tolist(),
                           mean_actual_after_first_2s=traces['actual_velocity'][trim:end,0].mean(axis=0).tolist(),
                           mae=np.abs(actual-np.asarray(cmd)).mean(axis=0).tolist()))
    save('velocity_summary.json', dict(phases=phases, frames=frames, fps=25))
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

