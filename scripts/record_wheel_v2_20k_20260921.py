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
from pxr import UsdGeom, Usd
import omni.usd
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_error_magnitude
import LeggedManip_Lab.tasks  # noqa: F401


def main():
    torch.set_num_threads(4)
    task = 'A2W-Z1-WBC-WheelV2-Play'
    cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs)
    assert a.num_envs == 1
    assert cfg.actions.wheel_vel.scale == 5.
    assert cfg.actions.wheel_vel.clip == {'.*': (-100., 100.)}
    assert cfg.scene.robot.actuators['wheels'].damping == .6
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == 'b723b1143b7c9c4a3b1bbe3bb1573c6584fb39a106724093c9c3d1eff3153db7'
    cfg.seed = a.seed
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    for ranges in [cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges]:
        ranges.lin_vel_x = (1., 1.)
        ranges.lin_vel_y = (0., 0.)
        ranges.ang_vel_z = (0., 0.)
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
        local = quat_apply_inverse(q, ee - link)
        mixed = torch.cat((local[:, :2] - command[:, :2], ee[:, 2:3] - command[:, 2:3]), -1)
        # Secondary metric: the exact target construction used by upstream debug visualization.
        delta = torch.zeros_like(ee)
        delta[:, :2] = command[:, :2]
        visual_goal = link + quat_apply(q, delta)
        visual_goal[:, 2] = command[:, 2]
        velocity = u.command_manager.get_command('base_velocity')
        assert torch.allclose(velocity, velocity.new_tensor([1.,0.,0.]).expand_as(velocity), atol=1e-6)
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
        'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'source_hashes': source_hashes, 'workspace_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'seed': a.seed, 'num_envs': a.num_envs, 'seconds_per_env': a.seconds,
        'step_dt': u.step_dt, 'steps': total, 'termination_terms': terms,
        'protocol': 'Fixed body velocity command [1,0,0], standing probability zero, heading command off. Native random EE pose goals and 20s episode resets retained. Native PLAY randomization and delays. 1-env qualitative video, not a 64-env benchmark.',
        'ee_native_definition': 'norm([link0-local x/y error, world z error]); NOT a pure world Euclidean norm when tilted.',
        'ee_visual_world_definition': 'World Euclidean error to upstream visualization target: link0 + R_link0*[command_x,command_y,0], then overwrite world z.',
        'velocity_definition': 'Per-frame body-frame XY Euclidean error and absolute yaw-rate error; NOT accumulated TensorBoard command metrics.',
    })
    writer = imageio.get_writer(str(a.output/'play.mp4'), fps=25, codec='libx264', quality=8, macro_block_size=1)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    cyan, orange = '#00e5ff', '#ff9b36'
    projections = []
    def annotate(im, target, actual, labels=True):
        d = ImageDraw.Draw(im)
        t = tuple(map(float,target)); e = tuple(map(float,actual))
        d.line([t,e],fill='black',width=6)
        d.line([t,e],fill='white',width=3)
        x,y=t
        d.polygon([(x,y-17),(x+17,y),(x,y+17),(x-17,y)],outline='black',width=6)
        d.polygon([(x,y-15),(x+15,y),(x,y+15),(x-15,y)],outline=cyan,width=3)
        d.line([(x-23,y),(x-7,y)],fill=cyan,width=2)
        d.line([(x+7,y),(x+23,y)],fill=cyan,width=2)
        x,y=e
        d.ellipse((x-9,y-9,x+9,y+9),outline='black',width=5)
        d.ellipse((x-7,y-7,x+7,y+7),outline=orange,width=3)
        if labels:
            for point,text,color,offset in [(t,'TARGET',cyan,(-105,-50)),(e,'ACTUAL EE',orange,(25,27))]:
                anchor=(max(8,min(im.width-150,point[0]+offset[0])),max(125,min(im.height-82,point[1]+offset[1])))
                d.line([point,(anchor[0]+35,anchor[1]+12)],fill=color,width=2)
                box=d.textbbox(anchor,text,font=small)
                d.rectangle((box[0]-5,box[1]-4,box[2]+5,box[3]+4),fill=(20,25,34))
                d.text(anchor,text,font=small,fill=color)
    frames = 0
    with torch.inference_mode():
        for i in range(total):
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise RuntimeError('Nonfinite actions')
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            if i % 2 == 0:
                base = robot.data.root_pos_w[0].cpu().numpy()
                u.sim.set_camera_view(eye=base+np.array([2.5,-3.5,1.8]), target=base+np.array([.35,0,.15]))
                rgb = raw.render()
                im = Image.fromarray(rgb)
                # Match current rendered state (including post-reset state), not stale pre-reset telemetry.
                ee_now = robot.data.body_pos_w[0,pose.body_idx]
                link_now = robot.data.body_pos_w[0,pose.link0_idx]
                q_now = robot.data.body_quat_w[0,pose.link0_idx]
                delta = torch.zeros((1,3),device=u.device)
                delta[0,:2] = pose.command[0,:2]
                goal_now = link_now + quat_apply(q_now.unsqueeze(0),delta)[0]
                goal_now[2] = pose.command[0,2]
                camera = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(cfg.viewer.cam_prim_path))
                c = camera.GetCamera(Usd.TimeCode.Default())
                c.transform = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                vp = np.asarray(c.frustum.ComputeViewMatrix()) @ np.asarray(c.frustum.ComputeProjectionMatrix())
                def project(pt):
                    clip = np.r_[pt.detach().cpu().numpy(),1.] @ vp
                    assert clip[3]>0, 'Marker behind camera'
                    xy = clip[:2]/clip[3]
                    return np.array([(xy[0]+1)*640,(1-xy[1])*360])
                target_px, actual_px = project(goal_now), project(ee_now)
                assert np.isfinite(target_px).all() and np.isfinite(actual_px).all()
                projections.append([*target_px,*actual_px])
                annotate(im,target_px,actual_px)
                # Live magnified inset: same frame and same projected geometry, no schematic offset.
                center = (target_px+actual_px)/2
                span = max(95.,float(np.max(np.abs(target_px-actual_px)))+65.)
                left,top = int(center[0]-span/2),int(center[1]-span/2)
                side=int(span)
                zoom=Image.fromarray(rgb).crop((left,top,left+side,top+side)).resize((280,280))
                annotate(zoom,(target_px-[left,top])*280/side,(actual_px-[left,top])*280/side,False)
                im.paste(zoom,(980,175))
                draw = ImageDraw.Draw(im)
                draw.rectangle((978,173,1262,457),outline='white',width=2)
                draw.rectangle((978,140,1262,173),fill=(20,25,34))
                draw.text((990,146),'LIVE TARGET / EE CLOSE-UP',font=small,fill='white')
                draw.rectangle((0,0,1280,118), fill=(20,25,34))
                v = traces['actual_velocity'][i,0]
                draw.text((15,10), f'LeggedManip_Lab | A2W + Z1 | WheelV2 20K | t={(i+1)*u.step_dt:.2f}s',font=font,fill='white')
                draw.text((15,44), f'Command vx=1.00 vy=0 yaw=0 | actual vx={v[0]:.2f} vy={v[1]:.2f} yaw={v[2]:.2f}',font=font,fill='white')
                fail = int((traces['terminated'][:i+1] & ~traces['timeout'][:i+1]).sum())
                visual_error = float((goal_now-ee_now).norm().item()*100)
                draw.text((15,78), f'Random EE goals | world marker distance={visual_error:.1f} cm | failure resets={fail}',font=font,fill='white')
                draw.rectangle((0,670,1280,720),fill=(20,25,34))
                draw.text((15,681),'CYAN DIAMOND: TARGET',font=small,fill=cyan)
                draw.text((330,681),'ORANGE RING: ACTUAL EE',font=small,fill=orange)
                draw.text((690,681),'WHITE LINE: POSITION ERROR',font=small,fill='white')
                if bool(dones[0]):
                    draw.text((15,130),'EPISODE RESET (native timeout / termination)',font=small,fill='yellow')
                writer.append_data(np.asarray(im))
                if frames in [0,125,375,625]: im.save(a.output/f'preview_{frames:04d}.jpg')
                frames += 1
            if (i + 1) % 500 == 0:
                state = dict(stage='EVALUATING', seed=a.seed, steps=i+1, total_steps=total,
                             mean_native_ee_cm=float(traces['ee_native_m'][:i+1].mean()*100))
                save('status.json', state)
                print('PROGRESS', json.dumps(state), flush=True)
    assert index == total
    writer.close()
    np.save(a.output/'projected_markers.npy',np.asarray(projections))
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
    save('summary.json', summary)
    save('velocity_summary.json', dict(command=[1.,0.,0.], mean_actual_velocity=traces['actual_velocity'].mean(axis=(0,1)).tolist(),
        post_2s_mean_actual_velocity=traces['actual_velocity'][round(2/u.step_dt):].mean(axis=(0,1)).tolist(),
        vx_mae=float(np.abs(traces['actual_velocity'][:,:,0]-1).mean()), frames=frames,fps=25))
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
