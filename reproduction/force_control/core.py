"""Force-only mechanisms ported from Improbable-AI/learning-compliance.

Upstream commit c760e1d74ad165d3c069d4f57ab5d066f6a41eb6, MIT (vendor/LICENSE).
Public force convention is EXTERNAL FORCE ON ROBOT, expressed in base-yaw axes.
Thus quasi-static force exerted by robot on environment has the opposite sign.
No inverse dynamics/gravity compensation is performed here.
"""
from dataclasses import dataclass, asdict
import torch

@dataclass
class ForceConfig:
    history_length: int = 10  # scripts/train.py, not paper's H=30
    mode_interval_s: float = 10.0
    force_probability: float = 0.5
    free_probability: float = 0.2  # source freed_envs: BOTH spring and command off
    force_limit_n: float = 70.0
    kp_range: tuple = (25.0, 400.0)
    kd_proportion: float = 0.1  # source overrides independently sampled [3,10]
    ramp_range_s: tuple = (1.0, 3.0)
    hold_s: float = 1.0
    cycle_range_s: tuple = (3.5, 9.0)
    force_sigma: float = 1.0 / 50.0
    axis_reward_weight: float = 3.0
    command_scale: float = 0.01
    anchor_mode: str = 'position_command'

    def __post_init__(self):
        if not 0 <= self.force_probability <= 1 or not 0 <= self.free_probability <= 1:
            raise ValueError('Probabilities must be in [0,1]')
        if self.force_limit_n <= 0 or self.history_length < 1 or self.hold_s < 0:
            raise ValueError('Invalid force/history/timing parameters')
        for bounds in [self.kp_range, self.ramp_range_s, self.cycle_range_s]:
            if not 0 < bounds[0] <= bounds[1]: raise ValueError('Invalid positive range')
        if self.anchor_mode not in ('position_command', 'entry_pose'):
            raise ValueError('Unknown spring anchor mode')


def yaw_angle(q_wxyz):
    w,x,y,z=q_wxyz.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def yaw_rotate(v, yaw, inverse=False):
    if inverse: yaw=-yaw
    c,s=yaw.cos(),yaw.sin();x,y,z=v.unbind(-1)
    return torch.stack((c*x-s*y,s*x+c*y,z),-1)


def force_reward(external_force_yaw, command_yaw, force_mode, sigma=1/50):
    """Exact upstream _reward_ee_force_{x,y,z}; no norm/squared-error substitution."""
    return torch.exp(-sigma*(external_force_yaw-command_yaw).abs())*force_mode[:,None]


def force_magnitude_penalty(external_force_world, force_mode, axis):
    """Upstream optional x/y force penalties (inactive in its published recipe)."""
    if axis not in (0, 1):
        raise ValueError('Upstream defines optional magnitude penalties for x/y only')
    return external_force_world[:, axis].abs()*force_mode


def spring_force(anchor_w, ee_pos_w, ee_vel_w, kp, kd, active, limit=70.):
    """Upstream clipped Kp*(x_set-x) - Kd*v_world; three axis clipping."""
    return (kp*(anchor_w-ee_pos_w)-kd*ee_vel_w).clamp(-limit,limit)*active[:,None]


def force_envelope(age, ramp, hold):
    """Ramp up, hold, ramp down, zero. Correct elapsed-time end boundary."""
    return torch.minimum(age/ramp,(2*ramp+hold-age)/ramp).clamp(0,1)


class ForceState:
    """Vectorized per-env state. No global episode clock or simulator dependencies."""
    def __init__(self,n,device,cfg=None):
        self.cfg=cfg or ForceConfig();self.n=n;self.device=device
        self.mode=torch.zeros(n,device=device,dtype=torch.bool)
        self.free=torch.zeros_like(self.mode)
        self.target=torch.zeros(n,3,device=device)
        self.command=torch.zeros_like(self.target)
        self.age=torch.zeros(n,device=device)
        self.ramp=torch.ones_like(self.age)
        self.cycle=torch.ones_like(self.age)
        self.kp=torch.ones(n,1,device=device);self.kd=self.kp.clone()
        self.external_w=torch.zeros_like(self.target)
        self.external_yaw=torch.zeros_like(self.target)
        self.anchor_yaw=torch.zeros_like(self.target)
        self.applied_substeps=torch.zeros(n,dtype=torch.long,device=device)

    def _uniform(self,ids,bounds,cols=None):
        shape=(len(ids),) if cols is None else (len(ids),cols)
        return bounds[0]+torch.rand(shape,device=self.device)*(bounds[1]-bounds[0])

    def sample_cycle(self,ids):
        self.age[ids]=0
        self.free[ids]=torch.rand(len(ids),device=self.device)<self.cfg.free_probability
        self.target[ids]=self._uniform(ids,(-self.cfg.force_limit_n,self.cfg.force_limit_n),3)
        self.target[ids]*=(~self.free[ids])[:,None]
        self.ramp[ids]=self._uniform(ids,self.cfg.ramp_range_s)
        self.cycle[ids]=torch.maximum(self._uniform(ids,self.cfg.cycle_range_s),2*self.ramp[ids]+self.cfg.hold_s)
        self.command[ids]=0

    def reset(self,ids,ee_pos_w,base_pos_w,base_quat_w):
        self.mode[ids]=torch.rand(len(ids),device=self.device)<self.cfg.force_probability
        self.kp[ids]=self._uniform(ids,self.cfg.kp_range,1)
        self.kd[ids]=self.cfg.kd_proportion*self.kp[ids]
        self.sample_cycle(ids)
        self.anchor_yaw[ids]=yaw_rotate(ee_pos_w[ids]-base_pos_w[ids],yaw_angle(base_quat_w[ids]),inverse=True)
        self.external_w[ids]=0;self.external_yaw[ids]=0;self.applied_substeps[ids]=0

    def advance(self,dt):
        self.age+=dt
        ids=torch.where(self.age>=self.cycle)[0]
        if len(ids):self.sample_cycle(ids)
        self.command[:]=self.target*force_envelope(self.age,self.ramp,self.cfg.hold_s)[:,None]*self.mode[:,None]

    def apply(self,anchor_w,pos_w,vel_w,base_quat_w):
        self.external_w[:]=spring_force(anchor_w,pos_w,vel_w,self.kp,self.kd,self.mode&~self.free,self.cfg.force_limit_n)
        self.external_yaw[:]=yaw_rotate(self.external_w,yaw_angle(base_quat_w),inverse=True)
        self.applied_substeps+=1
        return self.external_w

    def state_dict(self):
        return {'config':asdict(self.cfg),'tensors':{k:v.clone() for k,v in vars(self).items() if isinstance(v,torch.Tensor)}}

    def load_state_dict(self,state):
        if state['config'] != asdict(self.cfg):raise ValueError('Force config mismatch')
        for k,v in state['tensors'].items():
            dest=getattr(self,k)
            if dest.shape!=v.shape:raise ValueError('Force state shape mismatch')
            dest.copy_(v)


def impedance_external_command(position_yaw,velocity_yaw,target_yaw,kp,kd,limit=70.):
    """Paper impedance in OUR source force-on-robot sign convention.

    Desired robot-on-world force is Kp*(target-position)-Kd*velocity.
    Negate it for the upstream external-force command. Inputs must be estimates,
    not simulator privileged state. Clamp matches training command envelope.
    """
    return (kp*(position_yaw-target_yaw)+kd*velocity_yaw).clamp(-limit,limit)
