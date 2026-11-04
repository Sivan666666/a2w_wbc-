"""A2W/PhysX boundary. Learning configuration lives in upstream.py unchanged.

Task differences: 22 actions, flat terrain, EE commands, delayed PD and base
contact. Reward coefficients remain Precision's; imagined torque/acceleration
use the Go2 analytic-surrogate design, never unconstrained learned outputs.
"""
import math
import torch
from supermimic.adapter import Adapter
from supermimic.core import qapply, qinv, tilt_quat, push_history
from . import state as st


class Commands:
    """Stable method for Dynamo; avoid ephemeral dict.__getitem__ callbacks."""
    def __init__(self,values):self.values=values
    def get_command(self,name):
        return self.values[:,:3] if name=='base_velocity' else self.values[:,3:]


def yaw(q):
    return torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2].square()+q[:,3].square()))


def rotate_inverse(v, heading):
    c, s = heading.cos(), heading.sin()
    return torch.stack((c*v[:,0]+s*v[:,1], -s*v[:,0]+c*v[:,1], v[:,2]), -1)


def heading_tilt_quat(g):
    # R_y(pitch) R_x(roll), NOT shortest gravity-aligning rotation (which
    # generally has a different Euler yaw). Required for yaw-frame velocities.
    roll=torch.atan2(-g[:,1],-g[:,2])/2
    pitch=torch.atan2(g[:,0],g[:,1:].square().sum(-1).sqrt().clamp_min(1e-6))/2
    cr,sr,cp,sp=roll.cos(),roll.sin(),pitch.cos(),pitch.sin()
    return torch.stack((cp*cr,cp*sr,sp*cr,-sp*sr),-1)


def pose_delta(p, n):
    xy = rotate_inverse(torch.cat((n[:,:2]-p[:,:2], torch.zeros_like(p[:,:1])), -1), p[:,2])
    angle = n[:,2]-p[:,2]
    return torch.cat((xy[:,:2], torch.atan2(angle.sin(), angle.cos())[:,None]), -1)


class Bridge(Adapter):
    def __init__(self, env):
        super().__init__(env)
        self.static_dim = self.static.shape[1]
        assert self.context().shape[1] == st.CONTEXT_DIM
        self.limb_ids = []
        for name in ('thigh', 'calf', 'wheel'):
            ids, _ = self.robot.find_bodies([f'{leg}_{name}' for leg in ('FR','FL','RR','RL')], preserve_order=True)
            assert len(ids) == 4
            self.limb_ids.append(ids)
        self.default_active = self.default[self.ids[:18]]
        ki, di, li, effort = [None]*22, [None]*22, [None]*22, [None]*22
        offset = self.static_dim
        for act in self.robot.actuators.values():
            indices = list(range(self.robot.num_joints)) if isinstance(act.joint_indices, slice) else list(act.joint_indices)
            width = len(indices)
            for j, native in enumerate(indices):
                if native in self.ids:
                    a = self.ids.index(native)
                    ki[a], di[a], li[a] = offset+j, offset+width+j, offset+2*width
                    effort[a] = float(act.effort_limit[0,j])
            offset += 2*width+1
        assert all(v is not None for v in ki+di+li+effort)
        self.ki, self.di, self.li = ki, di, li
        self.effort = torch.tensor(effort, device=env.device)
        self.fall_angle = env.cfg.terminations.bad_orientation.params['limit_angle']
        # Exact collision box on URDF base; other fixed links are not base_contact.
        import xml.etree.ElementTree as ET
        box = ET.parse(env.cfg.scene.robot.spawn.asset_path).getroot().find("link[@name='base']/collision")
        assert box.find('geometry/box') is not None
        self.base_half = torch.tensor([float(x)/2 for x in box.find('geometry/box').get('size').split()],device=env.device)
        self.base_origin = torch.tensor([float(x) for x in box.find('origin').get('xyz').split()],device=env.device)
        assert box.find('origin').get('rpy', '0 0 0') == '0 0 0'
        # Preserve original capture (98 measured values) and additionally snapshot
        # aligned state and pose BEFORE automatic reset.
        original = env.reward_manager.compute
        def compute(dt):
            r = original(dt)
            with torch.no_grad():
                self.captured['aligned'] = self.aligned_state().clone()
                self.captured['pose'] = self.pose().clone()
            return r
        env.reward_manager.compute = compute

    def pose(self):
        d = self.robot.data
        return torch.cat((d.root_pos_w[:,:2], yaw(d.root_quat_w)[:,None]), -1)

    def proxy(self,state,action,previous,commands):
        result=super().proxy(state,action,previous,commands)
        result.command_manager=Commands(commands)
        return result

    def aligned_state(self):
        d = self.robot.data
        heading = yaw(d.root_quat_w)
        limbs = []
        for ids in self.limb_ids:
            rel = d.body_pos_w[:,ids] - d.root_pos_w[:,None]
            limbs.append(rotate_inverse(rel.flatten(0,1), heading[:,None].expand(-1,4).flatten()).reshape(-1,12))
        return torch.cat((d.root_pos_w[:,2:3], d.projected_gravity_b,
                          rotate_inverse(d.root_lin_vel_w,heading), d.root_ang_vel_b,
                          d.joint_pos[:,self.ids[:18]]-self.default_active,
                          d.joint_vel[:,self.ids], *limbs),-1)

    def legacy_state(self, s, torque=None, acceleration=None):
        z = torch.zeros_like(s[:,:1])
        tau = z.expand(-1,22) if torque is None else torque
        acc = z.expand(-1,22) if acceleration is None else acceleration
        return torch.cat((s[:,:4], qapply(qinv(heading_tilt_quat(s[:,st.GRAV])),s[:,st.V_YAW]),
                          s[:,st.W_B],s[:,st.Q]+self.default_active,s[:,st.QD],tau,acc,z.expand(-1,4)),-1)

    def analytic_aux(self, s0, s1, a, previous, context):
        # Endpoint PD approximation as in Go2; respect A2W's delayed final target,
        # mixed position/velocity action map and per-episode randomized gains.
        delayed = torch.where(context[:,self.li] >= self.env.cfg.decimation, previous, a)
        q = s1[:,st.Q]+self.default_active
        target = (self.default_active+.25*delayed[:,:18]).clamp(-10,10)
        wheel_target = (5*delayed[:,18:]).clamp(-100,100)
        kp, kd = context[:,self.ki], context[:,self.di]
        torque = torch.cat((kp[:,:18]*(target-q)-kd[:,:18]*s1[:,st.QD][:,:18],
                            kd[:,18:]*(wheel_target-s1[:,st.QD][:,18:])), -1)
        return torque.clamp(-self.effort,self.effort), (s1[:,st.QD]-s0[:,st.QD])/self.env.step_dt

    def fall_prob(self, s):
        tilt = torch.acos((-s[:,3]).clamp(-1+1e-4,1-1e-4))
        fall = torch.sigmoid((tilt-self.fall_angle)/math.radians(5.))
        # Analytic base-ground clearance, analogous to Go2 smooth hit geometry.
        q = tilt_quat(s[:,st.GRAV])
        center = s[:,0]+qapply(q,self.base_origin.expand(len(s),-1))[:,2]
        extent = (torch.nn.functional.normalize(s[:,st.GRAV],dim=-1).abs()*self.base_half).sum(-1)
        contact = torch.sigmoid(-(center-extent)/.03)
        return 1-(1-fall)*(1-contact)

    def critic_input(self, s, commands, context, previous):
        # Privileged physical state + task goal + physics context, as in Go2.
        # Previous action is necessary for A2W delayed actuators.
        return torch.cat((s,commands,context,previous),-1)

    def imagined_step(self, s, previous, history, commands, context, actor, model):
        x = self.critic_input(s,commands,context,previous)
        a = actor(history)
        nxt, _ = model(s,a,previous,context,torch.zeros((len(s),77),device=s.device))
        tau, acc = self.analytic_aux(s,nxt,a,previous,context)
        proxy = self.proxy(self.legacy_state(nxt,tau,acc),a,previous,commands)
        r = self.reward(proxy)
        d = self.fall_prob(nxt)
        obs = self.observation(proxy,'policy',noise=True)
        return nxt,a,push_history(history,obs),r,d,x

    @torch.no_grad()
    def validate_aligned(self):
        self.validate()
        s = self.aligned_state()
        legacy = self.legacy_state(s)
        torch.testing.assert_close(legacy[:,:50],self.state()[:,:50],atol=3e-4,rtol=1e-3)
        proxy = self.proxy(legacy,self.env.action_manager.action,self.env.action_manager.prev_action,self.commands())
        torch.testing.assert_close(self.observation(proxy,'policy'),self.observation(self.env,'policy'),atol=3e-4,rtol=1e-3)
        return {'state_dim':86,'context_dim':766,'actor_dim':234,'action_dim':22,
                'critic_dim':884,'observation_and_state_parity':'PASS',
                'reward_surrogate':'analytic endpoint delayed PD + control-period velocity difference; not exact substep PhysX torque/acc'}
