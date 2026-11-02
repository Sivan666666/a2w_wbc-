"""Isaac Lab adapter: originals remain authoritative for rewards/observations.

Only this process is instrumented. No installed Isaac Lab files are changed.
"""
from types import SimpleNamespace as NS
import torch
from .core import *


class Scene(dict):
    @property
    def sensors(self):
        return self


class Adapter:
    def __init__(self, env):
        self.env = env
        self.robot = r = env.scene['robot']
        from LeggedManip_Lab.assets.a2w_z1.a2w_z1_articulation_cfg import ACTIVE_JOINTS, POSITION_JOINTS
        self.names = ACTIVE_JOINTS
        self.ids, _ = r.find_joints(ACTIVE_JOINTS, preserve_order=True)
        assert len(self.ids) == 22 and len(set(self.ids)) == 22
        self.default = r.data.default_joint_pos[0].clone()
        self.limits = r.data.soft_joint_pos_limits[0].clone()
        self.fk = ArmFK(env.cfg.scene.robot.spawn.asset_path, POSITION_JOINTS).to(env.device)
        self.ee = r.find_bodies('end_effector')[0][0]
        self.link0 = r.find_bodies('link0')[0][0]
        self.sensor = env.scene['contact_forces']
        self.wheel_ids, _ = self.sensor.find_bodies(['FR_wheel', 'FL_wheel', 'RR_wheel', 'RL_wheel'], preserve_order=True)
        assert len(self.wheel_ids) == 4
        self.obs_cfg = env.observation_manager._group_obs_term_cfgs
        self.reward_cfg = [env.reward_manager.get_term_cfg(n) for n in env.reward_manager.active_terms]
        self.reward_names = list(env.reward_manager.active_terms)
        allowed = {'position_command_error_exp', 'orientation_command_error', 'track_lin_vel_xy_exp',
                   'track_ang_vel_z_exp', 'base_height_tracking', 'lin_vel_z_l2', 'ang_vel_xy_l2',
                   'joint_torques_l2', 'joint_torques_max', 'joint_acc_l2', 'action_rate_l2',
                   'joint_power', 'joint_deviation_l1', 'flat_orientation_l2', 'joint_pos_limits'}
        assert all(c.func.__name__ in allowed for c in self.reward_cfg), [(n,c.func.__name__) for n,c in zip(self.reward_names,self.reward_cfg)]
        # Startup-only randomized physical properties. Retained on GPU; no PhysX
        # CPU property reads in the training loop. Gains/delays are read separately.
        view = r.root_physx_view
        static = [view.get_masses(), view.get_inertias(), view.get_coms(), view.get_material_properties()]
        self.static = torch.cat([v.to(env.device).reshape(env.num_envs, -1) for v in static], -1)
        for act in r.actuators.values():
            assert act.cfg.max_delay <= env.cfg.decimation, 'Longer delays need a longer action queue'
        self.captured = None
        original = env.reward_manager.compute
        self.original_reward_compute = original

        def compute(dt):
            reward = original(dt)
            # Runs AFTER physics/termination and BEFORE resets/command resampling.
            with torch.no_grad():
                self.captured = dict(target=self.state().clone(),
                                     terminated=env.reset_terminated.clone(), done=env.reset_buf.clone(),
                                     reward=reward.clone())
            return reward
        env.reward_manager.compute = compute

    def context(self):
        parts = [self.static]
        for act in self.robot.actuators.values():
            parts.extend((act.stiffness, act.damping, act.positions_delay_buffer.time_lags[:, None].float()))
        return torch.cat(parts, -1)

    def state(self):
        d = self.robot.data
        return torch.cat((d.root_pos_w[:, 2:3], d.projected_gravity_b, d.root_lin_vel_b,
                          d.root_ang_vel_b, d.joint_pos[:, self.ids[:18]], d.joint_vel[:, self.ids],
                          d.applied_torque[:, self.ids], d.joint_acc[:, self.ids],
                          self.sensor.data.net_forces_w[:, self.wheel_ids].norm(dim=-1)), -1)

    def commands(self):
        return torch.cat((self.env.command_manager.get_command('base_velocity'),
                          self.env.command_manager.get_command('ee_pose')), -1).clone()

    def remaining(self):
        e = self.env
        times = [e.command_manager.get_term(n).time_left for n in ('base_velocity', 'ee_pose')]
        times.append((e.max_episode_length-e.episode_length_buf)*e.step_dt)
        return torch.stack(times).min(0).values[:, None]

    def proxy(self, state, action, previous, commands):
        b, device = len(state), state.device
        root_q = tilt_quat(state[:, G])
        root_p = torch.cat((torch.zeros_like(state[:, :2]), state[:, Z]), -1)
        local = self.fk(state[:, Q])
        body_p = root_p[:, None].expand(-1, self.robot.num_bodies, -1).clone()
        body_q = root_q[:, None].expand(-1, self.robot.num_bodies, -1).clone()
        for name, idx in [('link0', self.link0), ('end_effector', self.ee)]:
            p, q = local[name]
            body_p[:, idx] = root_p+qapply(root_q, p)
            body_q[:, idx] = qmul(root_q, q)
        joint_pos = self.default.expand(b, -1).clone()
        joint_pos[:, self.ids[:18]] = state[:, Q]
        def native(x):
            result = torch.zeros((b, self.robot.num_joints), device=device)
            result[:, self.ids] = x
            return result
        data = NS(root_pos_w=root_p, root_quat_w=root_q, root_lin_vel_b=state[:, V],
                  root_ang_vel_b=state[:, W], projected_gravity_b=state[:, G],
                  body_pos_w=body_p, body_quat_w=body_q, joint_pos=joint_pos,
                  joint_vel=native(state[:, QD]), applied_torque=native(state[:, TAU]),
                  joint_acc=native(state[:, ACC]), default_joint_pos=self.default.expand(b,-1),
                  default_joint_vel=torch.zeros_like(joint_pos), soft_joint_pos_limits=self.limits.expand(b,-1,-1))
        robot = NS(data=data, find_joints=self.robot.find_joints, find_bodies=self.robot.find_bodies)
        forces = torch.zeros((b, len(self.sensor.body_names), 3), device=device)
        forces[:, self.wheel_ids, 2] = state[:, CONTACT]
        scene = Scene(robot=robot, contact_forces=NS(data=NS(net_forces_w=forces)))
        command_dict = {'base_velocity': commands[:, :3], 'ee_pose': commands[:, 3:]}
        return NS(scene=scene, device=device, num_envs=b,
                  action_manager=NS(action=action, prev_action=previous),
                  command_manager=NS(get_command=command_dict.__getitem__))

    def observation(self, proxy, group, noise=False):
        parts = []
        for cfg in self.obs_cfg[group]:
            x = cfg.func(proxy, **cfg.params)
            if noise and cfg.noise is not None:
                assert hasattr(cfg.noise, 'n_min'), 'Unsupported noise; do not silently change it'
                x = x + torch.empty_like(x).uniform_(cfg.noise.n_min, cfg.noise.n_max)
            if cfg.clip is not None:
                x = x.clamp(*cfg.clip)
            if cfg.scale is not None:
                x = x*cfg.scale
            parts.append(x)
        return torch.cat(parts, -1)

    def reward(self, proxy):
        return sum(c.weight*c.func(proxy, **c.params) for c in self.reward_cfg)*self.env.step_dt

    def metrics(self, proxy):
        d = proxy.scene['robot'].data
        command = proxy.command_manager.get_command('ee_pose')
        rel = qapply(qinv(d.body_quat_w[:, self.link0]), d.body_pos_w[:, self.ee]-d.body_pos_w[:, self.link0])
        err = torch.cat((rel[:, :2]-command[:, :2], d.body_pos_w[:, self.ee, 2:3]-command[:, 2:3]), -1).norm(dim=-1)
        desired = d.body_pos_w[:, self.link0]+qapply(d.body_quat_w[:, self.link0], torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), -1))
        desired = torch.cat((desired[:, :2], command[:, 2:3]), -1)
        return {'ee_native_m': err.mean(), 'ee_world_m': (d.body_pos_w[:, self.ee]-desired).norm(dim=-1).mean(),
                'vel_xy_mps': (d.root_lin_vel_b[:, :2]-proxy.command_manager.get_command('base_velocity')[:, :2]).norm(dim=-1).mean()}

    @torch.no_grad()
    def validate(self):
        """Verify same-state reward, FK, instantaneous obs and history layout."""
        e, d = self.env, self.robot.data
        s = self.state(); a = e.action_manager.action; prev = e.action_manager.prev_action
        proxy = self.proxy(s, a, prev, self.commands())
        real_rel = qapply(qinv(d.root_quat_w), d.body_pos_w[:, self.ee]-d.root_pos_w)
        fk_pos, fk_q = self.fk(s[:, Q])['end_effector']
        real_q = qmul(qinv(d.root_quat_w), d.body_quat_w[:, self.ee])
        pos_error = (real_rel-fk_pos).norm(dim=-1).max().item()
        quat_error = torch.minimum((real_q-fk_q).norm(dim=-1), (real_q+fk_q).norm(dim=-1)).max().item()
        assert pos_error < 2e-4 and quat_error < 2e-4, ('FK', pos_error, quat_error)
        errors = {}
        for name, cfg in zip(self.reward_names, self.reward_cfg):
            real = cfg.func(e, **cfg.params); other = cfg.func(proxy, **cfg.params)
            torch.testing.assert_close(other, real, atol=5e-4, rtol=1e-3, msg=name)
            errors[name] = (real-other).abs().max().item()
        for group in ('policy', 'critic'):
            real = self.observation(e, group)
            other = self.observation(proxy, group)
            # EE quaternion may differ by sign; canonical root rotation cancels
            # from relative orientation so require same sign too for actor/critic.
            torch.testing.assert_close(other, real, atol=3e-4, rtol=1e-3, msg=group)
        return dict(fk_position_max_m=pos_error, fk_quaternion_max=quat_error, reward_errors=errors,
                    actor_dim=234, actions=22, critic_dim=self.observation(proxy,'critic').shape[-1],
                    state_dim=STATE, context_dim=self.context().shape[-1])
