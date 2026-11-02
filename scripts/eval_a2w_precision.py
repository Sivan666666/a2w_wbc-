"""World-point and trajectory tests, isolated from native training code."""
import argparse
import hashlib
import json
import math
import subprocess
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['point', 'trajectory'], required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--trials', type=int, default=10)
p.add_argument('--duration', type=float, default=20)
p.add_argument('--num_envs', type=int, default=64)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
def save(name, data):
    (a.output/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
save('status.json', dict(stage='STARTING'))
app = AppLauncher(a).app
import importlib.metadata
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate, quat_from_euler_xyz, quat_error_magnitude
import LeggedManip_Lab.tasks
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_wbc import UniformPoseWBCCommand
from a2w_precision_metrics import summarize


class Schedule:
    trial = 0
    time = 0.
    anchor_position = None
    anchor_quat = None
    def __init__(self):
        rng = np.random.default_rng(a.seed)
        shape = (a.trials, a.num_envs)
        self.points = np.stack([rng.uniform(.724, 1.024, shape), rng.uniform(-.35, .35, shape),
                                rng.uniform(.1, 1., shape)], -1)
        self.centers = np.stack([np.full(shape, .824), rng.uniform(-.08,.08,shape),
                                 rng.uniform(.58,.68,shape)], -1)
        self.kind = (np.arange(a.num_envs)[None, :] + np.arange(a.trials)[:, None]) % 3

    def goal(self, time):
        if a.mode == 'point':
            return self.points[self.trial].copy()
        # 5 s initial acquisition, then two smooth cycles over 15 s.
        s = np.clip((time-5)/15, 0, 1)
        phase = 4*np.pi*(10*s**3 - 15*s**4 + 6*s**5)
        g = self.centers[self.trial].copy()
        kind = self.kind[self.trial]
        g[kind == 0, 1] += .08*np.sin(phase)
        g[kind == 0, 2] += .08*np.cos(phase)
        g[kind == 1, 1] += .10*np.sin(phase)
        g[kind == 1, 2] += .06*np.sin(2*phase)
        g[kind == 2, 0] -= .10*np.cos(phase)
        return g

schedule = Schedule()


class WorldCommand(UniformPoseWBCCommand):
    def _resample_command(self, env_ids):
        self._update_command()

    def world_goal(self, time):
        target = torch.as_tensor(schedule.goal(time), dtype=torch.float32, device=self.device)
        local = torch.zeros_like(target)
        local[:,:2] = target[:,:2]
        local[:,0] -= .174
        initial = schedule.anchor_position + quat_apply(schedule.anchor_quat, local)
        unit_z = torch.zeros_like(local)
        unit_z[:,2] = 1
        world_z_axis = quat_apply(schedule.anchor_quat, unit_z)
        delta_z = (target[:,2]+self.env.scene.env_origins[:,2]-initial[:,2])/world_z_axis[:,2]
        return initial + world_z_axis*delta_z[:,None]

    def _update_command(self):
        link = self.robot.data.body_pos_w[:, self.link0_idx]
        q = self.robot.data.body_quat_w[:, self.link0_idx]
        if schedule.anchor_position is None:
            schedule.anchor_position = link.clone()
            schedule.anchor_quat = q.clone()
        self.target_w = self.world_goal(schedule.time)
        local = quat_apply_inverse(q, self.target_w-link)
        self.pose_command[:, :2] = local[:, :2]
        self.pose_command[:, 2] = self.target_w[:, 2]
        # Fixed world orientation for each trial; no wrist reorientation along path.
        reference = schedule.points[schedule.trial] if a.mode == 'point' else schedule.centers[schedule.trial]
        delta = torch.as_tensor(reference-[.174,0,0], dtype=torch.float32, device=self.device)
        delta[:,2] -= schedule.anchor_position[:,2]-self.env.scene.env_origins[:,2]
        pitch = (-torch.atan2(delta[:,2], delta[:,:2].norm(dim=-1))).clamp(-3.14/4,3.14/3)
        yaw = torch.atan2(delta[:,1],delta[:,0]).clamp(-3.14/4,3.14/3)
        self.target_quat_w = quat_mul(schedule.anchor_quat, quat_from_euler_xyz(torch.zeros_like(yaw), pitch, yaw))
        self.pose_command[:,3:] = quat_mul(quat_conjugate(q), self.target_quat_w)


def main():
    torch.set_num_threads(4)
    task = 'A2W-Z1-WBC-Play'
    cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs)
    cfg.seed = a.seed
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.ee_pose.debug_vis = cfg.commands.base_velocity.debug_vis = False
    cfg.commands.ee_pose.class_type = WorldCommand
    cfg.commands.base_velocity.rel_standing_envs = 1.
    # Native reset samples velocity before its standing-mask update. Make even
    # that first observation zero; curriculum reset also copies limit_ranges.
    for ranges in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
        ranges.lin_vel_x = ranges.lin_vel_y = ranges.ang_vel_z = (0., 0.)
    # Explicit synchronized trial resets; retain failure terminations during trial.
    cfg.episode_length_s = a.duration + .1
    cfg.log_dir = str(a.output)
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    agent.seed = a.seed
    agent.device = a.device
    dump_yaml(str(a.output/'env.yaml'), cfg)
    dump_yaml(str(a.output/'agent.yaml'), agent)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
    u = env.unwrapped
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=a.device)
    runner.load(str(a.checkpoint))
    policy = runner.get_inference_policy(device=u.device)
    pose = u.command_manager.get_term('ee_pose')
    robot = u.scene['robot']
    S = round(a.duration/u.step_dt)
    shape = (a.trials, S, a.num_envs)
    data = {k: np.empty(shape,np.float32) for k in ('error_m','issued_error_m','orientation_rad',
                'tilt_deg','phase_time_s','mapping_residual_m')}
    for name,n in [('reference_w',3),('issued_target_w',3),('ee_position_w',3),('root_position_w',3),
                   ('link_position_w',3),('link_quat_w',4),('command',7)]:
        data[name] = np.empty((*shape,n),np.float32)
    data['terminated'] = np.empty(shape,bool)
    data['kind'] = schedule.kind
    data['initial_root_w'] = np.empty((a.trials,a.num_envs,3),np.float32)
    originals = u.reward_manager.compute
    step = 0
    def capture(dt):
        r = originals(dt)
        ee = robot.data.body_pos_w[:,pose.body_idx]
        reference = pose.world_goal(schedule.time)
        link = robot.data.body_pos_w[:,pose.link0_idx]
        q = robot.data.body_quat_w[:,pose.link0_idx]
        arrays = dict(error_m=(ee-reference).norm(dim=-1),issued_error_m=(ee-pose.target_w).norm(dim=-1),
          orientation_rad=quat_error_magnitude(robot.data.body_quat_w[:,pose.body_idx],pose.target_quat_w),
          tilt_deg=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))*180/math.pi,
          reference_w=reference,issued_target_w=pose.target_w,ee_position_w=ee,
          root_position_w=robot.data.root_pos_w,link_position_w=link,link_quat_w=q,command=pose.command)
        for k,v in arrays.items():
            arr=v.detach().cpu().numpy()
            if not np.isfinite(arr).all(): raise RuntimeError(f'Nonfinite {k}')
            data[k][schedule.trial,step]=arr
        data['terminated'][schedule.trial,step]=u.termination_manager.terminated.cpu().numpy()
        data['phase_time_s'][schedule.trial,step]=schedule.time
        return r
    u.reward_manager.compute = capture
    diff = subprocess.check_output(['git','diff','--binary'])
    (a.output/'workspace.diff').write_bytes(diff)
    sources = sorted(Path('source/LeggedManip_Lab/LeggedManip_Lab').rglob('*.py'))
    save('source_hashes.json', {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    save('manifest.json', dict(mode=a.mode,seed=a.seed,num_envs=a.num_envs,trials=a.trials,duration=a.duration,
      workspace_diff_sha256=hashlib.sha256(diff).hexdigest(),
      checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      metrics_sha256=hashlib.sha256(Path(__file__).with_name('a2w_precision_metrics.py').read_bytes()).hexdigest(),
      commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      protocol='World-fixed goals; zero base velocity, floating base. Native policy/actions/PD/reward/failure terminations and PLAY DR retained. Per-trial full reset; timeout moved to duration+0.1s to avoid unintended boundary reset. Command generation replaced ONLY in this evaluation process.',
      coordinates='Recompute command XY=R_link0^-1(goal_world-link0_world).xy and command Z=goal_world.z each observation step; orientation=conj(q_link0)*q_world. No IK, no actuator compensation, no base position controller.',
      timing='Primary distance to reference at end of physics step; also save distance to zero-order-held target issued at step start. No lag correction.',
      point_workspace='Target anchored ONCE to each trial actual initial link0 position/orientation (including reset XY/yaw randomization), then fixed in world. Initial link0 XY: x .55...85, y -.35...35; world height .1..1. Not IK-feasibility filtered. Stored points include a nominal .174 x offset removed before anchoring.',
      trajectory='5s hold first point (excluded ONLY from moving-phase metric), then 15s two cycles with quintic phase; circle yz r=.08, figure8 amplitudes y=.10 z=.06, reach_retract x amplitude=.10. Center x=.824,y in +/-.08,z .58...68. Orientation fixed per trial. No bottle/contact/grasp success test.',
      orientation='Roll zero, pitch/yaw point from initial mount toward goal/trajectory center, clamped as native command, rotated into world by initial link0 orientation and held fixed; no random orientation offsets.'))
    save('targets.json',dict(points=schedule.points.tolist(),centers=schedule.centers.tolist(),kind=schedule.kind.tolist()))
    with torch.inference_mode():
        for trial in range(a.trials):
            schedule.trial=trial
            schedule.time=0.
            schedule.anchor_position = schedule.anchor_quat = None
            obs,_=env.reset()
            policy.reset(torch.ones(a.num_envs,dtype=torch.bool,device=u.device))
            assert obs['policy'].shape == (a.num_envs,234)
            data['initial_root_w'][trial]=robot.data.root_pos_w.cpu().numpy()
            for step in range(S):
                # Check the command adapter against its intended goal before physics.
                local=quat_apply_inverse(robot.data.body_quat_w[:,pose.link0_idx],pose.target_w-robot.data.body_pos_w[:,pose.link0_idx])
                residual=torch.cat((local[:,:2]-pose.command[:,:2],pose.target_w[:,2:3]-pose.command[:,2:3]),-1).norm(dim=-1)
                data['mapping_residual_m'][trial,step]=residual.cpu().numpy()
                assert residual.max()<1e-5
                assert not u.command_manager.get_command('base_velocity').any()
                schedule.time=(step+1)*u.step_dt
                action=policy(obs)
                if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite action')
                obs,_,dones,_=env.step(action)
                policy.reset(dones)
            state=dict(stage='EVALUATING',mode=a.mode,seed=a.seed,completed_trials=trial+1,total_trials=a.trials)
            save('status.json',state)
            print('PROGRESS',json.dumps(state),flush=True)
    np.savez_compressed(a.output/'trace.npz',**data)
    result=summarize(data,a.mode,u.step_dt)
    result['seed']=a.seed
    result['mapping_max_residual_m']=float(data['mapping_residual_m'].max())
    save('summary.json',result)
    print('SUMMARY',json.dumps(result),flush=True)
    env.close()
    save('status.json',dict(stage='COMPLETE',seed=a.seed,mode=a.mode))

try:
    main()
except BaseException:
    save('status.json',dict(stage='FAILED',traceback=traceback.format_exc()))
    raise
finally:
    app.close()
